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
  region = var.region

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
  for_each = toset(["model", "backend"])

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

# docker-compose.yaml lives in the repo, not in a heredoc inside this module. Putting
# it here rather than inlining it with file() means editing the stack is an
# upload plus a restart, not an instance replacement.
resource "aws_s3_bucket" "deploy" {
  bucket        = "socrates-deploy-${data.aws_caller_identity.current.account_id}"
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
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.gpu.id]
  iam_instance_profile   = aws_iam_instance_profile.gpu.name

  # An EIP is the stable address; without one the public IP changes on every
  # stop/start, and you will be stopping this constantly to avoid the bill.
  associate_public_ip_address = !var.use_elastic_ip

  root_block_device {
    volume_size = var.root_volume_gb # DLAMI + an 8 GB checkpoint needs room
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/bootstrap.sh", {
    region    = var.region
    registry  = local.registry
    image_tag = var.image_tag
    bucket    = aws_s3_bucket.deploy.id
  })

  tags = { Name = "socrates-gpu" }

  lifecycle {
    # a newer DLAMI publishes every few days; do not replace the box for it
    ignore_changes = [ami]
  }
}

resource "aws_eip" "gpu" {
  count    = var.use_elastic_ip ? 1 : 0
  instance = aws_instance.gpu.id
  domain   = "vpc"
}
