#!/usr/bin/env bash
#
# Benchmark our engine(s) against vLLM on the same box, same load, same metrics.
#
#   ./infra/prod/compare-vllm.sh                          # instance from terraform output
#   ./infra/prod/compare-vllm.sh i-0abc123
#   RATE=2 SEED=0 ./infra/prod/compare-vllm.sh
#   ENGINES="cont paged" TAG=dev ./infra/prod/compare-vllm.sh   # day 8's three-way run
#
# ENGINES runs each of our engines in turn (ENGINE=cont|paged on the model tier),
# then vLLM, all in one session, so "paged beat cont" is measured against a cont
# run from the same box ten minutes earlier rather than against last week's
# number. TAG picks the image tag for model and backend: `dev` is what
# kernel-on-instance.sh builds, the default is whatever /opt/socrates/.env says.
#
# Sequential, not side by side. An L4 has 24 GB; ours holds ~13 and vLLM is told
# to take 85% of the card, so running both at once would measure contention
# rather than either engine.
#
# socrates.service is stopped for the duration and the stack is driven directly
# from the shipped compose file (/tmp/src, which kernel-on-instance.sh and
# build-on-instance.sh unpack), because the copy in /opt/socrates is re-fetched
# from S3 on every service start and may predate ENGINE. The service is started
# again at the end, which puts production back exactly as it was.
#
# The comparison is only worth anything if the workload is identical, which is
# why the load generator lives in one place (backend/server.py) and speaks both
# protocols, rather than pitting our harness against vLLM's own benchmark script.
# Same prompts, same seeded Poisson arrivals, same greedy sampling, same token
# ceiling, thinking mode off on both, and every metric computed from the
# timestamps of tokens as they arrive.
set -uo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-management}"
RATE="${RATE:-2}"
SEED="${SEED:-0}"
BIN_S="${BIN_S:-1.0}"
ENGINES="${ENGINES:-cont}"
TAG="${TAG:-}"
LOG="${LOG:-/tmp/socrates-compare.log}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# Prepended to every instance-side script that touches compose. -p pins the
# project name so volumes (weights) and container names match what
# socrates.service creates. Exported, not prefixed: SSM may run these under
# dash, where VAR=x before a function call is not reliably passed on.
PRE=". /opt/socrates/.env
CF=/tmp/src/docker-compose.yaml; [ -f \"\$CF\" ] || CF=/opt/socrates/docker-compose.yaml
export IMAGE_TAG=${TAG:-\$IMAGE_TAG}
dc() { docker compose -p socrates -f \"\$CF\" --env-file /opt/socrates/.env \"\$@\"; }"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

as_json() { python3 -c 'import json,sys; print(json.dumps([sys.stdin.read()]))'; }

ssm_send() {
  local payload
  payload=$(printf '%s' "$1" | as_json)
  aws ssm send-command --profile "$PROFILE" --region "$REGION" \
    --instance-ids "$IID" --document-name AWS-RunShellScript \
    --timeout-seconds 3600 \
    --cli-input-json "{\"Parameters\":{\"commands\":$payload}}" \
    --query Command.CommandId --output text
}

ssm_status() {
  aws ssm get-command-invocation --profile "$PROFILE" --region "$REGION" \
    --command-id "$1" --instance-id "$IID" --query Status --output text 2>/dev/null || echo Pending
}

ssm_output() {
  aws ssm get-command-invocation --profile "$PROFILE" --region "$REGION" \
    --command-id "$1" --instance-id "$IID" --query StandardOutputContent \
    --output text 2>/dev/null || true
}

ssm_run() {                       # ssm_run "<shell>" "<label>" [max-seconds]
  local cid st waited=0 max="${3:-1800}"
  cid=$(ssm_send "$1") || { say "$2: send failed"; return 1; }
  while :; do
    st=$(ssm_status "$cid")
    case "$st" in
      Success) ssm_output "$cid"; return 0 ;;
      Failed|Cancelled|TimedOut)
        say "$2: $st"; ssm_output "$cid" | tail -20 | tee -a "$LOG"; return 1 ;;
    esac
    [ $waited -ge "$max" ] && { say "$2: gave up after ${waited}s"; return 1; }
    sleep 15; waited=$((waited + 15))
  done
}

IID="${1:-}"
[ -z "$IID" ] && IID=$(cd "$HERE" && terraform output -raw instance_id)
: > "$LOG"
say "instance $IID   rate ${RATE}/s   seed $SEED   engines: $ENGINES   tag: ${TAG:-from .env}"

BENCH_BODY() {                    # BENCH_BODY <target> -> a JSON payload
  printf '{"target":"%s","rate":%s,"seed":%s,"bin_s":%s}' "$1" "$RATE" "$SEED" "$BIN_S"
}

say "stopping socrates.service; the stack is driven directly until the end"
# Belt and braces: if this script dies part-way -- laptop asleep, SSO session
# expired -- nothing on this side will ever start the service again. A timer on
# the host does it after SAFETY_S regardless; the whole comparison takes ~40
# minutes, and starting an already-running unit at the end is a no-op.
ssm_run "systemctl stop socrates
systemd-run --on-active=${SAFETY_S:-7200} --timer-property=AccuracySec=1s \
  /bin/systemctl start socrates.service
docker ps --format '{{.Names}}'" "stop service" 300 \
  | sed 's/^/    /' | tee -a "$LOG"

# ---- our engine(s) ---------------------------------------------------------

for E in $ENGINES; do
  say "starting the model tier with ENGINE=$E (warmup compiles its decode graphs)"
  ssm_run "$PRE
export ENGINE=$E
dc up -d --force-recreate model backend >/dev/null 2>&1
for i in \$(seq 1 90); do
  curl -s -m 5 http://localhost:8000/health | grep -q '\"engine\":\"$E\"' && break
  sleep 10
done
curl -s -m 5 http://localhost:8000/health; echo" "model ready ($E)" 1200 | sed 's/^/    /' | tee -a "$LOG"

  say "running the load against ENGINE=$E (a few minutes)"
  ssm_run "curl -sS -m 1700 -X POST http://localhost:8000/api/benchmark \
  -H 'Content-Type: application/json' -d '$(BENCH_BODY engine)' \
  -o /tmp/bench-$E.json
python3 -c \"
import json
d = json.load(open('/tmp/bench-$E.json'))
print('$E  %.1f tok/s over %.0fs, %d tokens, engine=%s' % (
    d['headline']['throughput_tps'], d['context']['wall_s'],
    d['context']['output_tokens'], d['context'].get('engine')))
\"" "bench $E" 1800 | sed 's/^/    /' | tee -a "$LOG"
done

# ---- hand the card to vLLM -------------------------------------------------

say "stopping our model tier and starting vLLM (pull is ~9 GB the first time)"
ssm_run "$PRE
set -x
dc stop model
dc --profile bench up -d vllm
sleep 5
docker ps --format '{{.Names}} {{.Status}}'" "vllm up" 1800 | sed 's/^/    /' | tee -a "$LOG"

say "waiting for vLLM to load the model and answer /v1/models"
ssm_run "for i in \$(seq 1 90); do
  if docker exec socrates-backend-1 python -c \"
import httpx,sys
try: sys.exit(0 if httpx.get('http://vllm:8100/v1/models', timeout=4).status_code==200 else 1)
except Exception: sys.exit(1)
\" 2>/dev/null; then echo READY; break; fi
  sleep 10
done
docker logs --tail 5 socrates-vllm-1 2>&1 | tail -5" "vllm ready" 1800 | sed 's/^/    /' | tee -a "$LOG"

say "running the identical load against vLLM"
ssm_run "curl -sS -m 1700 -X POST http://localhost:8000/api/benchmark \
  -H 'Content-Type: application/json' -d '$(BENCH_BODY vllm)' \
  -o /tmp/bench-vllm.json
python3 -c \"
import json
d = json.load(open('/tmp/bench-vllm.json'))
print('vllm  %.1f tok/s over %.0fs, %d tokens' % (
    d['headline']['throughput_tps'], d['context']['wall_s'],
    d['context']['output_tokens']))
\"" "bench vllm" 1800 | sed 's/^/    /' | tee -a "$LOG"

# ---- give the card back ----------------------------------------------------

say "stopping vLLM and handing the box back to socrates.service"
ssm_run "$PRE
set -x
dc --profile bench down
systemctl start socrates
sleep 10
docker ps --format '{{.Names}} {{.Status}}'" "restore" 900 | sed 's/^/    /' | tee -a "$LOG"

# ---- the answer ------------------------------------------------------------

say "comparison"
# The last engine in ENGINES is the candidate; with more than one, the first is
# the baseline it has to beat. bench-engine.json keeps the old name working for
# anything that read it.
set -- $ENGINES
FIRST="$1"; LAST="${@: -1}"
BASE_ARG=""
[ "$FIRST" != "$LAST" ] && BASE_ARG="--baseline /tmp/bench-$FIRST.json"
ssm_run "cp /tmp/bench-$LAST.json /tmp/bench-engine.json
SC=/tmp/src/infra/prod/show_compare.py
[ -f \"\$SC\" ] || SC=/opt/socrates/show_compare.py
python3 \"\$SC\" /tmp/bench-$LAST.json /tmp/bench-vllm.json $BASE_ARG" "compare" 300 | tee -a "$LOG"

say "waiting for production to finish warmup before leaving"
ssm_run "for i in \$(seq 1 60); do
  curl -s -m 5 http://localhost:8000/health | grep -q '\"device\":\"cuda\"' && break
  sleep 10
done
curl -s -m 5 http://localhost:8000/health; echo" "model back" 900 | sed 's/^/    /' | tee -a "$LOG"

say "done. results on the instance: $(for E in $ENGINES; do printf '/tmp/bench-%s.json ' "$E"; done)/tmp/bench-vllm.json"
