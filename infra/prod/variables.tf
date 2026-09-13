variable "subnet_id" {
  description = <<-EOT
    Which subnet to launch in. Null takes the first in the default VPC, which
    is a coin flip: a "default VPC" can contain private subnets, and a private
    one routes 0.0.0.0/0 through a NAT gateway. Outbound still works there, so
    ECR pulls and model downloads succeed and everything looks fine -- but
    inbound never arrives and the Elastic IP is decorative. Pin a subnet whose
    route table points at an internet gateway.
  EOT
  type        = string
  default     = null
}

variable "bucket_name" {
  description = "Deploy bucket: docker-compose.yaml plus the weights tarball. S3 names are globally unique."
  type        = string
  default     = "socrates-llm"
}

variable "aws_profile" {
  description = "Named profile from ~/.aws/config. Null uses the default credential chain. SSO profiles need `aws sso login --profile <name>` first."
  type        = string
  default     = null
}

variable "region" {
  description = "AWS region. Needs G-instance quota -- new accounts have zero."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = <<-EOT
    Qwen3-4B in bf16 is ~8 GB of weights plus ~1.2 GB of KV cache at
    MAX_BATCH=4, so ~10 GB. Decode is memory-bandwidth bound, so pick on
    bandwidth rather than FLOPs: g5.xlarge is an A10G at 600 GB/s, g6.xlarge
    an L4 at 300 GB/s for $0.20 less. g4dn is Turing and has no bf16 at all.
  EOT
  type        = string
  default     = "g5.xlarge"
}

variable "ami_id" {
  description = "Override the Deep Learning AMI lookup. Null means most recent."
  type        = string
  default     = null
}

variable "root_volume_gb" {
  description = "The DLAMI alone is ~100 GB before any model weights land."
  type        = number
  default     = 150
}

variable "image_tag" {
  description = "ECR tag to run. infra/push-to-ecr.sh tags with the git sha; pin that rather than latest for anything you care about."
  type        = string
  default     = "latest"
}

variable "api_port" {
  description = "Port the backend tier listens on. The model tier's 8080 is never exposed."
  type        = number
  default     = 8000
}

variable "api_ingress_cidrs" {
  description = <<-EOT
    CIDRs allowed to reach the backend directly. Leave empty and use SSM port
    forwarding instead -- see the ssm_port_forward output. Only populate this
    once a browser has to reach the box, and never with 0.0.0.0/0 while the
    API is unauthenticated.
  EOT
  type        = list(string)
  default     = []
}

variable "use_elastic_ip" {
  description = "Allocate a stable address. ~$3.60/mo, and required if a domain or a nip.io hostname points here."
  type        = bool
  default     = true
}

variable "ecr_keep_images" {
  description = "Images retained per repo. The model image is ~6 GB each."
  type        = number
  default     = 5
}
