#!/usr/bin/env bash
#
# Build the working tree on the GPU host as :dev images and run GPU work in them.
# Nothing is pushed and :latest is untouched, so production keeps its image.
#
#   ./infra/prod/kernel-on-instance.sh                       # tests + bench
#   STEPS="tests" ./infra/prod/kernel-on-instance.sh
#   STEPS="weights profile equiv" ./infra/prod/kernel-on-instance.sh i-0abc123
#   SKIP_BUILD=1 STEPS=bench ./infra/prod/kernel-on-instance.sh   # reuse :dev
#   tail -f /tmp/socrates-kernel.log
#
# Steps, run in the order given, each in its own container from the :dev image:
#   tests    pytest -m cuda             kernel vs reference, split-K edges, graphs
#   weights  pytest -m weights          paged engine on the real checkpoint, compiled
#   bench    bench/day_8/kernel_bench.py   paged kernel vs dense SDPA, per partition
#   profile  bench/day_5/profile_decode.py with ENGINE=paged
#   equiv    bench/day_8/equivalence.py    cont, then paged, then compare
#   control  dense engine with a 2048 read window, against both of equiv's runs
#
# The card is shared, so socrates.service is stopped for the run and started
# again afterwards -- the API is down meanwhile, as with profile-on-instance.sh.
# Results land in /tmp/prof on the instance and the summaries are echoed here.
set -euo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-management}"
BUCKET="${BUCKET:-socrates-llm}"
STEPS="${STEPS:-tests bench}"
LOG="${LOG:-/tmp/socrates-kernel.log}"
MAX_WAIT="${MAX_WAIT:-3400}"            # the build
RUN_WAIT="${RUN_WAIT:-10800}"           # watching the steps; they finish regardless
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
as_json() { python3 -c 'import json,sys; print(json.dumps([sys.stdin.read()]))'; }

ssm_send() {
  local payload
  payload=$(printf '%s' "$1" | as_json)
  aws ssm send-command --profile "$PROFILE" --region "$REGION" \
    --instance-ids "$IID" --document-name AWS-RunShellScript --timeout-seconds 3600 \
    --cli-input-json "{\"Parameters\":{\"commands\":$payload,\"executionTimeout\":[\"3600\"]}}" \
    --query Command.CommandId --output text
}
ssm_status() {
  aws ssm get-command-invocation --profile "$PROFILE" --region "$REGION" \
    --command-id "$1" --instance-id "$IID" --query Status --output text 2>/dev/null || echo Pending
}
ssm_output() {
  aws ssm get-command-invocation --profile "$PROFILE" --region "$REGION" \
    --command-id "$1" --instance-id "$IID" --query StandardOutputContent --output text 2>/dev/null || true
}
ssm_wait() {                                  # ssm_wait <cid> <label> <tail-file>
  local cid="$1" label="$2" tailfile="$3" waited=0 st t
  while :; do
    st=$(ssm_status "$cid")
    case "$st" in
      Success) say "$label: Success"; return 0 ;;
      Failed|Cancelled|TimedOut)
        say "$label: $st"; ssm_output "$cid" | tail -40 | tee -a "$LOG"; return 1 ;;
    esac
    if [ $waited -ge "$MAX_WAIT" ]; then say "$label: gave up after ${waited}s"; return 1; fi
    if [ $((waited % 60)) -eq 0 ] && [ $waited -gt 0 ]; then
      t=$(ssm_send "tail -2 $tailfile 2>/dev/null || echo '(no output yet)'")
      sleep 8
      ssm_output "$t" | sed 's/^/    | /' | tee -a "$LOG"
    fi
    sleep 20; waited=$((waited + 20))
  done
}

IID="${1:-}"
[ -z "$IID" ] && IID=$(cd "$HERE" && terraform output -raw instance_id)
: > "$LOG"
say "instance $IID   steps: $STEPS"

# ---- ship and build ----------------------------------------------------------

if [ -z "${SKIP_BUILD:-}" ]; then
  say "packaging the working tree"
  TARBALL=$(mktemp -t socrates-src).tar.gz
  tar --exclude='.git' --exclude='.venv' --exclude='node_modules' \
      --exclude='__pycache__' --exclude='bench/results' --exclude='.next' \
      --exclude='*.tfstate*' --exclude='.terraform' --exclude='*.so' \
      -czf "$TARBALL" -C "$REPO" .
  aws s3 cp "$TARBALL" "s3://$BUCKET/socrates-src.tar.gz" --profile "$PROFILE" --only-show-errors
  rm -f "$TARBALL"

  say "building model:dev and backend:dev on the instance (first kernel build pulls the CUDA toolkit)"
  BUILD=$(ssm_send "set -x
. /opt/socrates/.env
rm -rf /tmp/src && mkdir -p /tmp/src /tmp/prof && chmod 777 /tmp/prof
df -h / | tail -1
docker image prune -af --filter until=48h || true
docker builder prune -f --keep-storage=20GB || true
df -h / | tail -1
aws s3 cp s3://$BUCKET/socrates-src.tar.gz /tmp/src/src.tar.gz
cd /tmp/src && tar xzf src.tar.gz
docker build -f model/Dockerfile -t \$REGISTRY/model:dev . > /tmp/prof/build.log 2>&1; echo MODEL_BUILD=\$?
docker build -f backend/Dockerfile -t \$REGISTRY/backend:dev . >> /tmp/prof/build.log 2>&1; echo BACKEND_BUILD=\$?
grep -E 'kernel ok|nvcc: NVIDIA|Cuda compilation|error' /tmp/prof/build.log | tail -8
df -h / | tail -1")
  ssm_wait "$BUILD" "build" /tmp/prof/build.log
  ssm_output "$BUILD" | grep -vE '^\+ ' | sed 's/^/    /' | tee -a "$LOG"
  if ssm_output "$BUILD" | grep -qE "_BUILD=[^0]"; then
    say "FAILED: build returned non-zero; see /tmp/prof/build.log on the instance"
    exit 1
  fi
fi

# ---- run ---------------------------------------------------------------------

# One container per step: a crash in one cannot leave CUDA state for the next,
# and each step's peak memory is its own.
RUNC='IMAGE_TAG=dev docker compose -f /opt/socrates/docker-compose.yaml --env-file /opt/socrates/.env run --rm --no-deps -v /tmp/prof:/out'
CMDS=""
for s in $STEPS; do
  case "$s" in
    tests)   CMDS+="$RUNC model pytest -m cuda -q -rs > /tmp/prof/tests.log 2>&1; echo STEP_tests=\$?
tail -15 /tmp/prof/tests.log
" ;;
    weights) CMDS+="$RUNC -e SOCRATES_WEIGHTS=1 -e ENGINE=paged model pytest -m weights -q -rs -s > /tmp/prof/weights.log 2>&1; echo STEP_weights=\$?
grep -E 'graphs warm|passed|failed|Error' /tmp/prof/weights.log | tail -8
" ;;
    bench)   CMDS+="$RUNC model python bench/day_8/kernel_bench.py --out /out/kernel_bench.json > /tmp/prof/bench.log 2>&1; echo STEP_bench=\$?
grep -vE '^\s*$' /tmp/prof/bench.log | tail -22
" ;;
    profile) CMDS+="$RUNC -e ENGINE=paged -e PROF_OUT=/out model python bench/day_5/profile_decode.py > /tmp/prof/profile.log 2>&1; echo STEP_profile=\$?
grep -E 'steady-state|SUMMARY|rows live|engine ready|graphs warm' /tmp/prof/profile.log
grep -icE 'skipping cudagraphs' /tmp/prof/profile.log | sed s/^/cudagraph_skip_lines=/
" ;;
    equiv)   CMDS+="$RUNC model python bench/day_8/equivalence.py run --engine cont --out /out/eq_cont.json > /tmp/prof/equiv.log 2>&1; echo STEP_equiv_cont=\$?
$RUNC -e ENGINE=paged model python bench/day_8/equivalence.py run --engine paged --out /out/eq_paged.json >> /tmp/prof/equiv.log 2>&1; echo STEP_equiv_paged=\$?
$RUNC -e ENGINE=paged model python bench/day_8/equivalence.py run --engine paged --force /out/eq_cont.json --out /out/eq_paged_forced.json >> /tmp/prof/equiv.log 2>&1; echo STEP_equiv_forced=\$?
$RUNC model python bench/day_8/equivalence.py compare /out/eq_cont.json /out/eq_paged.json --forced /out/eq_paged_forced.json; echo STEP_equiv_compare=\$?
grep -E 'graphs warm|tokens ->' /tmp/prof/equiv.log
" ;;
    control) CMDS+="$RUNC -e WINDOW_BUCKET=2048 model python bench/day_8/equivalence.py run --engine cont --out /out/eq_cont2048.json > /tmp/prof/control.log 2>&1; echo STEP_control_run=\$?
$RUNC model python bench/day_8/equivalence.py compare /out/eq_cont.json /out/eq_cont2048.json; echo STEP_control_cont_vs_cont2048=\$?
$RUNC model python bench/day_8/equivalence.py compare /out/eq_cont2048.json /out/eq_paged.json; echo STEP_control_cont2048_vs_paged=\$?
" ;;
    *) say "unknown step: $s"; exit 2 ;;
  esac
done

say "NOTE: the API is down while this runs"
# The steps run on the instance as a transient systemd unit, not inside the SSM
# command. An SSM command that times out is killed mid-step, and a laptop that
# sleeps or loses its SSO session stops polling -- either way, a final
# `systemctl start socrates` in the same command would never run and production
# would stay down. The EXIT trap below runs however the script ends, and the
# unit does not care what happens on this side.
SCRIPT="#!/bin/bash
set +e
exec > /tmp/prof/summary.txt 2>&1
trap 'systemctl start socrates; echo RUN_DONE; touch /tmp/prof/run.done' EXIT
systemctl stop socrates
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
$CMDS"
LAUNCH=$(ssm_send "mkdir -p /tmp/prof && chmod 777 /tmp/prof
rm -f /tmp/prof/run.done /tmp/prof/summary.txt
cat > /tmp/prof/run.sh <<'SOCRATES_EOS'
$SCRIPT
SOCRATES_EOS
systemd-run --unit=socrates-kernel-\$(date +%s) --collect /bin/bash /tmp/prof/run.sh && echo LAUNCHED")
ssm_wait "$LAUNCH" "launch" /dev/null
ssm_output "$LAUNCH" | grep -q LAUNCHED || { say "FAILED to launch the run unit"; exit 1; }

waited=0
while :; do
  sleep 60; waited=$((waited + 60))
  p=$(ssm_send "test -f /tmp/prof/run.done && echo RUN_FINISHED; f=\$(ls -t /tmp/prof/*.log 2>/dev/null | head -1); [ -n \"\$f\" ] && printf '%s: %s\n' \"\$f\" \"\$(tail -c 200 \"\$f\" | tr '\n' ' ')\"")
  sleep 8
  out=$(ssm_output "$p")
  printf '%s\n' "$out" | grep -v RUN_FINISHED | sed 's/^/    | /' | tee -a "$LOG"
  case "$out" in *RUN_FINISHED*) break ;; esac
  if [ $waited -ge "$RUN_WAIT" ]; then
    say "stopped watching after ${waited}s; the unit keeps running and restarts the service itself"
    exit 1
  fi
done

say "--- results ---"
RES=$(ssm_send "cat /tmp/prof/summary.txt")
sleep 10
ssm_output "$RES" | sed 's/^/    /' | tee -a "$LOG"
SUMMARY=$(ssm_output "$RES")

say "waiting for the service to come back"
for _ in $(seq 1 30); do
  h=$(ssm_send "curl -s -m 10 http://localhost:8000/health")
  sleep 12
  out=$(ssm_output "$h")
  case "$out" in *'"device":"cuda"'*) say "service healthy: $out"; break ;; esac
  sleep 18
done

if printf '%s' "$SUMMARY" | grep -qE "STEP_[a-z_]+=[^0]"; then
  say "at least one step failed"
  exit 1
fi
say "done"
