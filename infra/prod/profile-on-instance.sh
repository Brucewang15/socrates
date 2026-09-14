#!/usr/bin/env bash
#
# Profile a decode step on the deployed GPU host and bring the service back.
#
#   ./infra/prod/profile-on-instance.sh                # instance from terraform
#   ./infra/prod/profile-on-instance.sh i-0abc123
#   tail -f /tmp/socrates-profile.log                  # watch from another shell
#
# The model container holds the whole card, so this stops it, runs
# bench/day_5/profile_decode.py as a one-off container, then starts the service
# again. Expect the API to be down for a few minutes.
#
# Leaves on the instance: /tmp/prof/out.log, trace.json (~114 MB), table_cpu.txt,
# table_cuda.txt, summary.json. Reduce the trace with bench/day_5/analyze_trace.py
# rather than trying to copy it off.
set -euo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-management}"
LOG="${LOG:-/tmp/socrates-profile.log}"
MAX_WAIT="${MAX_WAIT:-2400}"
HERE="$(cd "$(dirname "$0")" && pwd)"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
as_json() { python3 -c 'import json,sys; print(json.dumps([sys.stdin.read()]))'; }

ssm_send() {
  local payload
  payload=$(printf '%s' "$1" | as_json)
  aws ssm send-command --profile "$PROFILE" --region "$REGION" \
    --instance-ids "$IID" --document-name AWS-RunShellScript --timeout-seconds 3600 \
    --cli-input-json "{\"Parameters\":{\"commands\":$payload}}" \
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

IID="${1:-}"
[ -z "$IID" ] && IID=$(cd "$HERE" && terraform output -raw instance_id)

: > "$LOG"
say "instance $IID"
say "log $LOG"
say "NOTE: the API is down while this runs"

CID=$(ssm_send "set -x
mkdir -p /tmp/prof && chmod 777 /tmp/prof
cp /tmp/src/bench/day_5/profile_decode.py /tmp/prof/ 2>/dev/null || aws s3 cp s3://socrates-llm/profile_decode.py /tmp/prof/profile_decode.py
systemctl stop socrates
cd /opt/socrates && docker compose stop model
cd /opt/socrates && docker compose run --rm -v /tmp/prof:/out -e PROF_OUT=/out model python /out/profile_decode.py > /tmp/prof/out.log 2>&1; echo PROF_EXIT=\$?
systemctl start socrates
grep -E 'steady-state|SUMMARY|rows live|engine ready' /tmp/prof/out.log
grep -icE 'skipping cudagraphs' /tmp/prof/out.log | sed s/^/cudagraph_skip_lines=/
grep -oE 'Self CUDA time total: [0-9.]+ms' /tmp/prof/out.log | tail -1
grep -oE 'Self CPU time total: [0-9.]+ms' /tmp/prof/out.log | tail -1")
say "command $CID"

waited=0
while :; do
  st=$(ssm_status "$CID")
  case "$st" in
    Success) say "profiler: Success"; break ;;
    Failed|Cancelled|TimedOut) say "profiler: $st"; ssm_output "$CID" | tail -25 | tee -a "$LOG"; exit 1 ;;
  esac
  if [ $waited -ge "$MAX_WAIT" ]; then say "gave up after ${waited}s (still $st)"; exit 1; fi
  if [ $((waited % 60)) -eq 0 ] && [ $waited -gt 0 ]; then
    t=$(ssm_send "tail -2 /tmp/prof/out.log 2>/dev/null || echo '(loading model)'")
    sleep 8
    ssm_output "$t" | sed 's/^/    | /' | tee -a "$LOG"
  fi
  sleep 20
  waited=$((waited + 20))
done

say "--- results ---"
ssm_output "$CID" | grep -vE '^\+ |^$' | sed 's/^/    /' | tee -a "$LOG"

say "waiting for the service to come back"
for _ in $(seq 1 15); do
  h=$(ssm_send "curl -s -m 10 http://localhost:8000/health")
  sleep 12
  out=$(ssm_output "$h")
  case "$out" in *'"device":"cuda"'*) say "service healthy again"; break ;; esac
  sleep 18
done
say "done"
