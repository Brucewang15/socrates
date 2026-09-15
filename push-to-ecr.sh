#!/usr/bin/env bash
#
# Build the GPU-host images and push them to ECR. The frontend is not here:
# it deploys to Vercel from the repo, so it has no image.
#
#   ./push-to-ecr.sh                  # model and backend
#   ./push-to-ecr.sh model            # just one
#
#   AWS_REGION=us-east-1 ./push-to-ecr.sh
#
# Each image is tagged twice: with the git sha, which is what a deployment
# should actually pin, and with latest, which is for convenience only.
#
# --platform linux/amd64 is not optional on Apple Silicon. Docker would
# otherwise build arm64 images that fail on a g5 instance with an exec format
# error, and the failure surfaces at run time, not at push time.
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
TAG="$(git rev-parse --short HEAD)"
PREFIX="socrates"

TIERS=("$@")
[ ${#TIERS[@]} -eq 0 ] && TIERS=(model backend prometheus grafana)

cd "$(dirname "$0")"     # repo root; build contexts are relative to it

if ! git diff --quiet HEAD 2>/dev/null; then
  echo "warning: working tree is dirty, so tag ${TAG} will not match what is in it" >&2
fi

echo "registry ${REGISTRY}"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

for tier in "${TIERS[@]}"; do
  repo="${PREFIX}/${tier}"
  uri="${REGISTRY}/${repo}"

  # idempotent: create-repository fails loudly if the repo already exists
  aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repo" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$AWS_REGION" --repository-name "$repo" \
         --image-scanning-configuration scanOnPush=true >/dev/null

  echo "==> building ${repo}:${TAG}"
  docker build \
    --platform linux/amd64 \
    -f "$([ -f "${tier}/Dockerfile" ] && echo "${tier}/Dockerfile" || echo "monitoring/Dockerfile.${tier}")" \
    -t "${uri}:${TAG}" \
    -t "${uri}:latest" \
    .

  echo "==> pushing ${repo}"
  docker push "${uri}:${TAG}"
  docker push "${uri}:latest"
  echo "${uri}:${TAG}"
done
