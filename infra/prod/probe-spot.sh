#!/usr/bin/env bash
#
# Bring the GPU host back on the cheapest Spot pool that has capacity.
#
#   ./infra/prod/probe-spot.sh                      # default pool order below
#   POOLS="g6.xlarge:a g6.2xlarge:a" ./infra/prod/probe-spot.sh
#
# A pool is an (instance type, AZ) pair. Each is tried with a real
# `terraform apply`, because nothing short of RunInstances says whether Spot has
# capacity -- placement scores read 1/10 everywhere on 2026-10-22 while
# g6.2xlarge in 1a launched first time. A pool is abandoned after 4
# InsufficientInstanceCapacity errors (~15 s) or 4 minutes.
#
# Candidates are every type with a 24 GB GPU and <= 8 vCPU (the Spot quota),
# cheapest first by the prices seen on 2026-10-22. g5 is an A10G (sm_86) and g6
# an L4 (sm_89); the model image compiles the kernel for both. Ruled out: g4dn
# (T4, 16 GB, no bf16), g6f (fractional L4, at most 11.4 GB).
#
# On success, pin the pool that landed in terraform.tfvars (instance_type and
# subnet_id); otherwise the next plain `terraform apply` will try to move it.
set -u
cd "$(dirname "$0")"

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-management}"
POOLS="${POOLS:-g5.xlarge:c g6.xlarge:b g5.xlarge:d g5.xlarge:f g5.xlarge:a g5.xlarge:b
g6.xlarge:c g6.xlarge:d g5.2xlarge:c g6.xlarge:a g5.2xlarge:d g6.xlarge:f
g6.2xlarge:a g5.2xlarge:a g6.2xlarge:f g6.2xlarge:d g6.2xlarge:c g6.2xlarge:b}"

VPC=$(aws ec2 describe-vpcs --profile "$PROFILE" --region "$REGION" \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
subnet_for() {
  aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
    --filters Name=vpc-id,Values="$VPC" Name=availability-zone,Values="$1" \
              Name=map-public-ip-on-launch,Values=true \
    --query 'Subnets[0].SubnetId' --output text
}

for pool in $POOLS; do
  TYPE=${pool%%:*}; AZ=$REGION${pool##*:}
  SUBNET=$(subnet_for "$AZ")
  [ "$SUBNET" = "None" ] && { echo "no public subnet in $AZ, skipping"; continue; }
  echo "$(date -u +%H:%M:%SZ) trying $TYPE in $AZ ($SUBNET)"
  LOG=/tmp/probe-spot-$TYPE-$AZ.log; rm -f "$LOG.debug"
  TF_LOG=DEBUG TF_LOG_PATH="$LOG.debug" terraform apply -auto-approve -no-color -input=false \
    -var "instance_type=$TYPE" -var "subnet_id=$SUBNET" > "$LOG" 2>&1 &
  PID=$!
  for i in $(seq 1 48); do
    sleep 5
    kill -0 $PID 2>/dev/null || break
    n=$(grep -c InsufficientInstanceCapacity "$LOG.debug" 2>/dev/null); n=${n:-0}
    if [ "$n" -ge 4 ] || [ "$i" -eq 48 ]; then
      echo "  no capacity ($n errors), moving on"
      kill -INT $PID; wait $PID
      break
    fi
  done
  wait $PID 2>/dev/null
  if grep -q 'Apply complete' "$LOG"; then
    echo "LANDED $TYPE in $AZ: $(terraform output -raw instance_id)"
    echo "pin it in terraform.tfvars: instance_type = \"$TYPE\", subnet_id = \"$SUBNET\""
    exit 0
  fi
done
echo "NO_CAPACITY in any pool"
exit 1
