# One GPU instance running all three containers, plus the ECR repos that feed
# it. No VPC of its own: the default VPC has public subnets and an internet
# gateway already, and building one here would only add a NAT gateway bill.
#
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }

  # State is a local file until you uncomment this. That is fine for one
  # person on one laptop and wrong the moment it is two.
  # backend "s3" {
  #   bucket       = "socrates-tfstate"
  #   key          = "prod/terraform.tfstate"
  #   region       = "us-east-1"
  #   encrypt      = true
  #   use_lockfile = true
  # }
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "socrates"
      Env       = "prod"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# The NVIDIA driver and container toolkit are preinstalled here. Installing
# them from user_data instead is slow and fails in interesting ways.
data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

locals {
  registry = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}

# ------- ECR --------

resource "aws_ecr_repository" "tier" {
  for_each = toset(["model", "backend", "prometheus", "grafana"])

  name = "socrates/${each.key}"

  image_scanning_configuration {
    scan_on_push = true
  }

  force_delete = true
}

# Model images are ~6 GB each. Without this, every push is kept forever.
resource "aws_ecr_lifecycle_policy" "tier" {
  for_each   = aws_ecr_repository.tier
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last ${var.ecr_keep_images} images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = var.ecr_keep_images
      }
      action = { type = "expire" }
    }]
  })
}

# ------- S3 --------

# Holds docker-compose.yaml and nothing else: 1 KB saying which images to run.
# Weights are not here -- the container pulls them from HuggingFace, which is
# faster than any upload from a laptop and keeps this bucket trivial.
resource "aws_s3_bucket" "deploy" {
  bucket        = var.bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "deploy" {
  bucket                  = aws_s3_bucket.deploy.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "compose" {
  bucket = aws_s3_bucket.deploy.id
  key    = "docker-compose.yaml"
  source = "${path.module}/../../docker-compose.yaml"
  etag   = filemd5("${path.module}/../../docker-compose.yaml")
}

# ------- IAM --------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu" {
  name               = "socrates-gpu"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# SSM is what replaces SSH keys and an open port 22. ECR read is what lets
# user_data pull the images.
resource "aws_iam_role_policy_attachment" "managed" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
  ])

  role       = aws_iam_role.gpu.name
  policy_arn = each.value
}

data "aws_iam_policy_document" "deploy_read" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.deploy.arn}/*"]
  }
}

resource "aws_iam_role_policy" "deploy_read" {
  name   = "socrates-deploy-read"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.deploy_read.json
}

# Push, not just pull. The managed ReadOnly policy above covers
# GetAuthorizationToken and pulls; these five are what `docker push` needs.
#
# Why the instance needs it at all: building an image on the box only puts it in
# that box's local store, so it dies with the box -- and on Spot that is hours,
# not months. Pushing from a laptop instead means an emulated linux/amd64 build
# and a 6.3 GB upload over home broadband; from here it is native and on AWS's
# own network. Scoped to these two repositories rather than using
# AmazonEC2ContainerRegistryPowerUser, which would grant push to every repo in
# the account.
data "aws_iam_policy_document" "ecr_push" {
  statement {
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = [for r in aws_ecr_repository.tier : r.arn]
  }
}

resource "aws_iam_role_policy" "ecr_push" {
  name   = "socrates-ecr-push"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.ecr_push.json
}

resource "aws_iam_instance_profile" "gpu" {
  name = "socrates-gpu"
  role = aws_iam_role.gpu.name
}

# ------- security group --------

resource "aws_security_group" "gpu" {
  name        = "socrates-gpu"
  description = "socrates GPU host"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "ECR, Hugging Face, apt"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Empty by default: SSM port forwarding reaches the backend without opening
# anything. Set api_ingress_cidrs only once a browser has to reach it directly.
resource "aws_security_group_rule" "api" {
  count = length(var.api_ingress_cidrs) > 0 ? 1 : 0

  type              = "ingress"
  description       = "backend API"
  security_group_id = aws_security_group.gpu.id
  from_port         = var.api_port
  to_port           = var.api_port
  protocol          = "tcp"
  cidr_blocks       = var.api_ingress_cidrs
}

# -------- EC2 ---------

resource "aws_instance" "gpu" {
  ami                    = var.ami_id != null ? var.ami_id : data.aws_ami.gpu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id != null ? var.subnet_id : data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  # Deliberately not setting associate_public_ip_address: the default subnet
  # has map_public_ip_on_launch, so AWS reports true no matter what we ask for,
  # and the attribute forces replacement -- every apply would rebuild the box.
  # The EIP attaches on top and is what gives it a stable address.

  root_block_device {
    volume_size = var.root_volume_gb # DLAMI + an 8 GB checkpoint needs room
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  # Spot when var.use_spot, on-demand otherwise. Defaults inside a spot launch
  # are the sane ones: max price = the on-demand price, so you are never billed
  # more than on-demand, and interruption terminates rather than stops.
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []

    content {
      market_type = "spot"
    }
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/bootstrap.sh", {
    region    = var.region
    registry  = local.registry
    image_tag = var.image_tag
    bucket    = aws_s3_bucket.deploy.id
  })

  tags = { Name = "socrates-gpu" }

  # Without this, a capacity-starved pool does not fail -- the provider retries
  # InsufficientInstanceCapacity internally and apply appears to hang for
  # 20+ minutes. Five minutes is long enough to ride out a transient shortage
  # and short enough to tell you to pick another AZ or instance type.
  timeouts {
    create = "5m"
  }

  lifecycle {
    # a newer DLAMI publishes every few days; do not replace the box for it
    ignore_changes = [ami]

    # Stand the replacement up before tearing the old one down. Without this a
    # replacement destroys first, so a create that fails -- no GPU quota, no
    # spot capacity in the AZ -- leaves you with nothing running and an
    # unattached EIP. The old and new instances bill together for a few minutes.
    create_before_destroy = true
  }
}

resource "aws_eip" "gpu" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.gpu.id
  domain   = "vpc"
}
