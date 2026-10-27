#!/usr/bin/env bash
# Start a loki-worker on an AWS EC2 GPU instance and print how to add it to a pool.
#
#   tools/worker/aws/launch-ec2.sh [--type g6e.xlarge] [--region us-east-1] [--key my-keypair]
#                                  [--cidr 203.0.113.4/32] [--disk 200] [--preload retouch,segment]
#
# Needs the AWS CLI with credentials. The instance runs the Deep Learning Base
# OSS NVIDIA Driver GPU AMI (Ubuntu 22.04: driver, Docker and the NVIDIA
# container toolkit preinstalled); user-data starts the container with a fresh
# token and keeps the models on the root gp3 volume. Port 8080 is opened only
# to --cidr (default: this machine's public IP). Instance types:
#   g6e.xlarge  L40S 48 GB   every engine, models stay loaded side by side
#   g6.xlarge   L4 24 GB     every engine (models take turns)
#   p5.48xlarge 8x H100      one node, eight job slots
# Stop it with: aws ec2 terminate-instances --instance-ids <id>
set -euo pipefail

TYPE=g6e.xlarge
REGION=${AWS_REGION:-us-east-1}
KEY=""
CIDR=""
DISK=200
PRELOAD=""
IMAGE=ghcr.io/stevecastle/loki-worker:latest
NAME=loki-worker
while [ $# -gt 0 ]; do
  case "$1" in
    --type) TYPE=$2; shift 2 ;;
    --region) REGION=$2; shift 2 ;;
    --key) KEY=$2; shift 2 ;;
    --cidr) CIDR=$2; shift 2 ;;
    --disk) DISK=$2; shift 2 ;;
    --preload) PRELOAD=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$CIDR" ] || CIDR="$(curl -fsS https://checkip.amazonaws.com | tr -d '\n')/32"
TOKEN="lw_$(openssl rand -hex 24)"

AMI=$(aws ssm get-parameter --region "$REGION" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)

SG=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=group-name,Values=loki-worker \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --region "$REGION" --group-name loki-worker \
    --description "loki-worker API" --query GroupId --output text)
fi
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
  --protocol tcp --port 8080 --cidr "$CIDR" >/dev/null 2>&1 || true

USERDATA=$(cat <<EOF
#!/bin/bash
set -e
mkdir -p /opt/loki-models
docker run -d --name loki-worker --restart unless-stopped --gpus all -p 8080:8080 \
  -e LOKI_WORKER_TOKEN=$TOKEN -e LOKI_WORKER_PRELOAD=$PRELOAD -e LOKI_WORKER_NAME=$NAME \
  -v /opt/loki-models:/models $IMAGE
EOF
)

KEYARG=()
[ -n "$KEY" ] && KEYARG=(--key-name "$KEY")
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" "${KEYARG[@]}" \
  --security-group-ids "$SG" --user-data "$USERDATA" \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK,VolumeType=gp3,DeleteOnTermination=true}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "instance $ID starting ($TYPE in $REGION)" >&2
aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "the container needs a few minutes to pull the image (and to download models on first use)" >&2
echo "lokictl pool add-node aws http://$IP:8080 --token $TOKEN --name $ID --no-test"
