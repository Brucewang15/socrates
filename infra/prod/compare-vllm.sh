#!/usr/bin/env bash
#
# Benchmark our engine against vLLM on the same box, same load, same metrics.
#
#   ./infra/prod/compare-vllm.sh                  # instance from terraform output
#   ./infra/prod/compare-vllm.sh i-0abc123
#   RATE=2 SEED=0 ./infra/prod/compare-vllm.sh
#
# Sequential, not side by side. An L4 has 24 GB; ours holds ~13 and vLLM is told
# to take 85% of the card, so running both at once would measure contention
# rather than either engine. The backend stays up throughout -- it is the harness
# -- while the model tier is stopped for vLLM's turn and started again after.
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
LOG="${LOG:-/tmp/socrates-compare.log}"
COMPOSE="docker compose -f /opt/socrates/docker-compose.yaml --env-file /opt/socrates/.env"
HERE="$(cd "$(dirname "$0")" && pwd)"

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
say "instance $IID   rate ${RATE}/s   seed $SEED"

BENCH_BODY() {                    # BENCH_BODY <target> -> a JSON payload
  printf '{"target":"%s","rate":%s,"seed":%s,"bin_s":%s}' "$1" "$RATE" "$SEED" "$BIN_S"
}

# ---- our engine ------------------------------------------------------------

say "checking the stack is up and the model is loaded"
ssm_run "$COMPOSE up -d --no-recreate model backend >/dev/null 2>&1
for i in \$(seq 1 60); do
  curl -s -m 5 http://localhost:8000/health | grep -q '\"device\":\"cuda\"' && break
  sleep 10
done
curl -s -m 5 http://localhost:8000/health; echo" "model ready" 900 | sed 's/^/    /' | tee -a "$LOG"

say "running the load against ours (a few minutes)"
ssm_run "curl -sS -m 1700 -X POST http://localhost:8000/api/benchmark \
  -H 'Content-Type: application/json' -d '$(BENCH_BODY engine)' \
  -o /tmp/bench-engine.json
python3 -c \"
import json
d = json.load(open('/tmp/bench-engine.json'))
print('ours  %.1f tok/s over %.0fs, %d tokens' % (
    d['headline']['throughput_tps'], d['context']['wall_s'],
    d['context']['output_tokens']))
\"" "bench ours" 1800 | sed 's/^/    /' | tee -a "$LOG"

# ---- hand the card to vLLM -------------------------------------------------

say "stopping our model tier and starting vLLM (pull is ~9 GB the first time)"
ssm_run "set -x
$COMPOSE stop model
$COMPOSE --profile bench up -d vllm
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

say "stopping vLLM and bringing our model tier back"
ssm_run "set -x
$COMPOSE --profile bench stop vllm
$COMPOSE --profile bench rm -f vllm
$COMPOSE start model
sleep 10
docker ps --format '{{.Names}} {{.Status}}'" "restore" 900 | sed 's/^/    /' | tee -a "$LOG"

# ---- the answer ------------------------------------------------------------

say "comparison"
# The path has to be resolved on the instance. Written with escaped dollars for
# that reason: an unescaped $(...) here would run on the laptop, where the
# deployed source does not exist, and silently pick the wrong file.
ssm_run "SC=/tmp/src/infra/prod/show_compare.py
[ -f \"\$SC\" ] || SC=/opt/socrates/show_compare.py
python3 \"\$SC\" /tmp/bench-engine.json /tmp/bench-vllm.json" "compare" 300 | tee -a "$LOG"

say "waiting for our model tier to finish warmup before leaving"
ssm_run "for i in \$(seq 1 60); do
  curl -s -m 5 http://localhost:8000/health | grep -q '\"device\":\"cuda\"' && break
  sleep 10
done
curl -s -m 5 http://localhost:8000/health; echo" "model back" 900 | sed 's/^/    /' | tee -a "$LOG"

say "done. results on the instance: /tmp/bench-engine.json /tmp/bench-vllm.json"
