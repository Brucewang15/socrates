# socrates GPU host — start / stop runbook

State of the world as of the last session: the GPU instance is **terminated**,
everything else (EIP, ECR images, S3 compose, IAM, security group) is still
provisioned and costs a few dollars a month. Terraform state is the local file
`infra/prod/terraform.tfstate`.

All commands run from `infra/prod/` unless noted. The AWS profile is pinned in
`terraform.tfvars`, so no `--profile` flag is needed for Terraform itself.

## Start it

```sh
cd infra/prod
terraform apply
```

That is the whole thing. Terraform refreshes, sees the instance is gone, and
creates a replacement, then re-associates the Elastic IP. Expect:

| step | time |
|---|---|
| instance boots | ~15 s |
| pull the 6.2 GB model image from ECR | ~2 min |
| download ~7.5 GB of weights from Hugging Face | ~2 min |
| model loads onto the card | ~1 min |

So roughly **5–8 minutes from apply to first token**. The weights are *not*
preserved across a termination: the root volume has `delete_on_termination =
true`, so every start re-downloads them.

Confirm it is actually serving, and actually on the GPU:

```sh
INSTANCE=$(terraform output -raw instance_id)

aws ssm start-session --region us-east-1 --target "$INSTANCE"
# on the box:
#   nvidia-smi                       -> A10G, ~9.2 GB used once the model loads
#   docker ps                        -> socrates-model-1, socrates-backend-1
#   curl -s localhost:8000/health    -> {"status":"ok","model":{...,"device":"cuda",...}}
#   journalctl -u socrates -f        -> pull/boot progress
```

`device: cuda` in that health response is the check that matters. If it says
`cpu`, the container did not get the card — see the compose notes below.

## Stop it

It is a **Spot** instance from a *one-time* request, so it cannot be stopped —
only terminated. `aws ec2 stop-instances` will dry-run clean and then fail for
real, which is misleading.

Terminate it with the CLI, **not** `terraform destroy -target=aws_instance.gpu`:

```sh
aws ec2 terminate-instances --profile management --region us-east-1 \
  --instance-ids "$(terraform output -raw instance_id)"
```

Why not `-target`: `aws_eip.gpu` references the instance, so Terraform treats it
as a dependent and releases the Elastic IP too. You would come back on a
different address. Terminating outside Terraform leaves the EIP allocated and
the state mildly stale, which the next `terraform apply` reconciles by itself.

## What keeps costing money while it is off

| resource | approx |
|---|---|
| Elastic IP `3.91.129.162`, idle | ~$3.60/mo |
| ECR storage, ~6.3 GB model + 371 MB backend, 5 images kept | ~$1–3/mo |
| S3 `socrates-llm` (a 1 KB compose file) | pennies |

The root EBS volume is deleted with the instance, so it costs nothing while off.
To drop the last few dollars, release the EIP — but the address is then gone for
good and `3-91-129-162.nip.io` stops resolving to you.

## Non-obvious things that will bite you

**Spot capacity, not quota, is the usual failure.** On-demand G quota in this
account was 0; a request for 8 vCPUs was filed and was `CASE_OPENED` at the time
of writing. Check whether it landed:

```sh
aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-DB2E81BA --profile management --region us-east-1 \
  --query 'Quota.Value'
```

If it is ≥ 8, set `use_spot = false` in `terraform.tfvars` and apply. That gets
you an instance AWS cannot reclaim, and one that *can* be stopped and started
without re-downloading the weights.

**`g5.2xlarge`, not `g5.xlarge`.** The xlarge Spot pool was empty in every
us-east-1 AZ (placement score 1/10, `InsufficientInstanceCapacity` on a real
launch). The 2xlarge pool had capacity and is 8 vCPU — exactly the Spot quota
ceiling, so nothing else GPU-shaped can run alongside it. Survey before changing
type:

```sh
aws ec2 get-spot-placement-scores --profile management --region us-east-1 \
  --instance-types g5.xlarge g5.2xlarge g6.2xlarge \
  --target-capacity 1 --single-availability-zone --region-names us-east-1 \
  --query 'reverse(sort_by(SpotPlacementScores,&Score))[:5]'
```

**An apply that seems to hang for 20 minutes is an empty Spot pool.** The AWS
provider retries `InsufficientInstanceCapacity` internally and says nothing.
`timeouts { create = "5m" }` on the instance now bounds it, so you get a real
error instead of silence.

**`subnet_id` is pinned on purpose.** Leaving it null takes
`data.aws_subnets.default.ids[0]`, which is unordered — it wanted to move the
box from `us-east-1a` to `us-east-1d` on an unrelated apply. The pinned subnet
routes via `igw-01676b782684c2de4`, so the EIP actually works there.

**`create_before_destroy` is load-bearing.** Without it a replacement destroys
the old instance first, so a create that fails on capacity leaves you with
nothing running and a detached EIP. With it, the old box keeps serving until the
new one exists.

**Compose controls whether the GPU is used at all.** `docker-compose.yaml` needs
both of these, and shipped with them disabled:

```yaml
environment:
  DEVICE: cuda
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Terraform uploads that file to S3 and `socrates-up` re-pulls it on every start,
so editing it plus `systemctl restart socrates` is a deploy — no instance
replacement needed.

**The API is unauthenticated.** `api_ingress_cidrs` is empty and should stay
empty; reach the backend over SSM instead of opening port 8000:

```sh
aws ssm start-session --region us-east-1 --target "$(terraform output -raw instance_id)" \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}'
# then, on your laptop:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/chat \
  -H 'Content-Type: application/json' -d '{"prompt":"what is 2+2?"}'
```

Note the field is **`prompt`**, not `message`.

**Two Terraform states exist.** This local file, and another on a collaborator's
laptop from when the stack was first built. Whoever applies second fights the
first. The fix is the `backend "s3"` block in `main.tf`, currently commented out.
