#!/bin/bash
# Rendered by templatefile() in main.tf, so Terraform substitutes its own
# interpolations before the shell ever sees this file. A shell variable here
# has to be written with a doubled dollar sign.
#
# Runs once, as root, on first boot only -- cloud-init does not re-run it on
# reboot. Anything that must survive a reboot belongs in the systemd unit this
# installs, not in this script.
#
# It deliberately does almost nothing: the stack it runs is described by
# docker-compose.yaml in the repo, uploaded to S3 by Terraform. Editing that file and
# restarting the unit is a deploy; editing this one replaces the instance.
#
# Output lands in /var/log/cloud-init-output.log.
set -euxo pipefail

install -d -m 0755 /opt/socrates

# The Deep Learning AMI ships docker, the NVIDIA driver and the container
# toolkit. Only the compose plugin may be missing.
if ! docker compose version >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y docker-compose-plugin
fi

# compose reads .env from the directory holding the compose file.
#
# REGISTRY carries the repository namespace, not just the host: push-to-ecr.sh
# pushes to <host>/socrates/<tier>, and compose asks for $${REGISTRY}/<tier>.
# Locally the compose default is a bare "socrates", which resolves the same way.
cat > /opt/socrates/.env <<ENV
REGISTRY=${registry}/socrates
IMAGE_TAG=${image_tag}
ENV

# An ECR login token lasts 12 hours, and docker-compose.yaml may have changed since
# this boot, so both are refreshed on every start rather than only here.
cat > /usr/local/bin/socrates-up <<SH
#!/bin/bash
set -euo pipefail
aws s3 cp s3://${bucket}/docker-compose.yaml /opt/socrates/docker-compose.yaml
aws ecr get-login-password --region ${region} \
  | docker login --username AWS --password-stdin ${registry}
cd /opt/socrates
docker compose pull
exec docker compose up --remove-orphans
SH
chmod 0755 /usr/local/bin/socrates-up

cat > /etc/systemd/system/socrates.service <<'UNIT'
[Unit]
Description=socrates inference stack
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/socrates-up
ExecStop=/usr/bin/docker compose -f /opt/socrates/docker-compose.yaml down
Restart=always
RestartSec=10
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now socrates.service
