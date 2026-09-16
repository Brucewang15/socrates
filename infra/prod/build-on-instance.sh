#!/usr/bin/env bash
#
# Build the GPU images on the EC2 instance, push them to ECR, then deploy.
#
#   ./infra/prod/build-on-instance.sh                  # instance from terraform output
#   ./infra/prod/build-on-instance.sh i-0abc123        # or name one
#   tail -f /tmp/socrates-deploy.log                   # watch from another shell
#
# Why not push-to-ecr.sh from a laptop: that builds --platform linux/amd64 under
# QEMU on Apple Silicon and then pushes 6.3 GB up a home connection. Measured:
# the `uv sync` layer takes 101s natively on the instance and had not finished
# after 47 minutes locally, and the push is ~35 min at 3.2 MB/s up versus ~1 min
# from inside AWS. Same build, same registry, roughly 10 minutes instead of hours.
#
# Requires the instance role to have ECR push -- see aws_iam_role_policy.ecr_push
# in main.tf. Everything here is idempotent; re-run it after a spot reclaim.
set -euo pipefail

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-management}"
BUCKET="${BUCKET:-socrates-llm}"
LOG="${LOG:-/tmp/socrates-deploy.log}"
MAX_WAIT="${MAX_WAIT:-3000}"            # give up after 50 minutes
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

say() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# Shell -> SSM parameters is a quoting minefield; let python build the JSON.
as_json() { python3 -c 'import json,sys; print(json.dumps([sys.stdin.read()]))'; }

ssm_send() {                            # ssm_send "<shell>" -> CommandId
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

ssm_wait() {                            # ssm_wait <cid> <label> [tail-file]
  local cid="$1" label="$2" tailfile="${3:-}" waited=0 st
  while :; do
    st=$(ssm_status "$cid")
    case "$st" in
      Success) say "$label: Success"; return 0 ;;
      Failed|Cancelled|TimedOut) say "$label: $st"; ssm_output "$cid" | tail -30 | tee -a "$LOG"; return 1 ;;
    esac
    if [ $waited -ge "$MAX_WAIT" ]; then
      say "$label: giving up after ${waited}s (still $st)"
      return 1
    fi
    # mirror the instance-side log so there is something to watch
    if [ -n "$tailfile" ] && [ $((waited % 60)) -eq 0 ] && [ $waited -gt 0 ]; then
      local t
      t=$(ssm_send "tail -3 $tailfile 2>/dev/null || echo '(no output yet)'")
      sleep 8
      ssm_output "$t" | sed 's/^/    | /' | tee -a "$LOG"
    fi
    sleep 20
    waited=$((waited + 20))
  done
}

# ---- which instance ---------------------------------------------------------

IID="${1:-}"
if [ -z "$IID" ]; then
  IID=$(cd "$HERE" && terraform output -raw instance_id)
fi

: > "$LOG"
say "instance $IID  region $REGION  profile $PROFILE"
say "log $LOG"

ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
TAG=$(cd "$REPO" && git rev-parse --short HEAD)
say "registry $REGISTRY   tag $TAG"

if ! (cd "$REPO" && git diff --quiet HEAD); then
  say "WARNING: working tree is dirty, so tag $TAG will not match what is built"
fi

# ---- ship the source --------------------------------------------------------

say "packaging source"
TARBALL=$(mktemp -t socrates-src).tar.gz
tar --exclude='.git' --exclude='.venv' --exclude='node_modules' \
    --exclude='__pycache__' --exclude='bench/results' --exclude='.next' \
    --exclude='*.tfstate*' --exclude='.terraform' \
    -czf "$TARBALL" -C "$REPO" .
say "uploading $(du -h "$TARBALL" | cut -f1) to s3://$BUCKET/socrates-src.tar.gz"
aws s3 cp "$TARBALL" "s3://$BUCKET/socrates-src.tar.gz" --profile "$PROFILE" --only-show-errors
rm -f "$TARBALL"

say "waiting for SSM to answer on $IID"
for _ in $(seq 1 30); do
  cid=$(ssm_send "cloud-init status 2>/dev/null; test -f /opt/socrates/docker-compose.yaml && echo COMPOSE_PRESENT" || true)
  [ -n "${cid:-}" ] && ssm_wait "$cid" "ssm probe" >/dev/null 2>&1 && break
  sleep 15
done
ssm_output "$cid" | sed 's/^/    | /' | tee -a "$LOG"

# ---- build and push ---------------------------------------------------------

say "building both images on the instance (native amd64; ~7 min for model)"
BUILD=$(ssm_send "set -x
systemctl stop socrates || true
rm -rf /tmp/src && mkdir -p /tmp/src /tmp/prof && chmod 777 /tmp/prof
# Reclaim before building, not after: the DLAMI is ~100 GB of the 150 GB root
# volume, weights are another ~8, and every model build adds a 6.4 GB image that
# keeps its own git-sha tag -- so it is not dangling and never gets collected.
# Three builds in a day is enough to fill the disk, and a full root volume takes
# the SSM agent down with it: run-command then fails with exit 1 and no output,
# because the agent cannot write the script it was asked to run. Everything
# removed here is in ECR and pulls back on demand.
df -h / | tail -1
docker image prune -af --filter until=48h || true
docker builder prune -f --keep-storage=10GB || true
df -h / | tail -1
aws s3 cp s3://$BUCKET/socrates-src.tar.gz /tmp/src/src.tar.gz
cd /tmp/src && tar xzf src.tar.gz
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
cd /tmp/src
docker build -f model/Dockerfile -t $REGISTRY/socrates/model:$TAG -t $REGISTRY/socrates/model:latest . > /tmp/deploy.log 2>&1; echo MODEL_BUILD=\$?
docker build -f backend/Dockerfile -t $REGISTRY/socrates/backend:$TAG -t $REGISTRY/socrates/backend:latest . >> /tmp/deploy.log 2>&1; echo BACKEND_BUILD=\$?
for img in model backend; do
  for t in $TAG latest; do
    docker push $REGISTRY/socrates/\$img:\$t >> /tmp/deploy.log 2>&1
    echo PUSH_\${img}_\${t}=\$?
  done
done
df -h / | tail -1
tail -5 /tmp/deploy.log")
say "command $BUILD"
ssm_wait "$BUILD" "build+push" /tmp/deploy.log
ssm_output "$BUILD" | grep -E "BUILD=|PUSH_|/dev/" | sed 's/^/    /' | tee -a "$LOG"

# A build that failed must not be reported as a deployment: the tags stay on
# whatever built last, so the service would restart on the old image and look
# fine. Fail loudly instead.
if ssm_output "$BUILD" | grep -qE "(MODEL|BACKEND)_BUILD=[^0]|PUSH_[a-z_]*=[^0]"; then
  say "FAILED: a build or push returned non-zero -- see /tmp/deploy.log on the instance"
  say "not restarting the service; :latest still points at the previous build"
  exit 1
fi

# ---- verify it actually landed ----------------------------------------------

say "verifying ECR"
for repo in model backend; do
  aws ecr describe-images --repository-name "socrates/$repo" --profile "$PROFILE" \
    --region "$REGION" \
    --query "sort_by(imageDetails,&imagePushedAt)[-1].{Tags:imageTags,Pushed:imagePushedAt,GB:imageSizeInBytes}" \
    --output json | sed "s/^/    $repo /" | tee -a "$LOG"
done

# ---- deploy the normal way, so it survives a reboot -------------------------

say "starting socrates.service (pulls :latest from ECR)"
UP=$(ssm_send "systemctl start socrates; sleep 25; systemctl is-active socrates; docker ps --format '{{.Names}} {{.Image}}'")
ssm_wait "$UP" "service start"
ssm_output "$UP" | sed 's/^/    /' | tee -a "$LOG"

say "waiting for the model to load"
for i in $(seq 1 20); do
  H=$(ssm_send "curl -s -m 10 http://localhost:8000/health; echo; nvidia-smi --query-gpu=memory.used --format=csv,noheader")
  sleep 12
  out=$(ssm_output "$H")
  say "health: $(printf '%s' "$out" | tr '\n' ' ')"
  case "$out" in *'"device":"cuda"'*) say "model live on GPU"; break ;; esac
  sleep 20
done

say "done. ECR now serves this build, so a spot reclaim only needs: terraform apply"
