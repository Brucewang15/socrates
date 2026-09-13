output "instance_id" {
  description = "Instance id, for the SSM commands below."
  value       = aws_instance.gpu.id
}

output "public_ip" {
  description = "Null when use_elastic_ip is false and the instance is stopped."
  value       = var.use_elastic_ip ? aws_eip.gpu[0].public_ip : aws_instance.gpu.public_ip
}

output "public_dns" {
  description = "AWS-assigned hostname. Fine for curl over plain HTTP; useless for a browser on an HTTPS page, because no CA will issue a certificate for amazonaws.com."
  value       = aws_instance.gpu.public_dns
}

output "ecr_repositories" {
  description = "Push targets. infra/push-to-ecr.sh creates these too -- same names, whichever runs first wins."
  value       = { for k, v in aws_ecr_repository.tier : k => v.repository_url }
}

output "ssm_shell" {
  description = "A shell on the box with no SSH key and no open port 22."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.gpu.id}"
}

output "ssm_port_forward" {
  description = "Reach the backend from your laptop with no public IP, no ingress rule and no domain. Swap both port numbers for 8080 to hit the model tier directly."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.gpu.id} --document-name AWS-StartPortForwardingSession --parameters '{\"portNumber\":[\"${var.api_port}\"],\"localPortNumber\":[\"${var.api_port}\"]}'"
}

output "boot_log" {
  description = "Where user_data's output lands. First place to look when nothing is listening."
  value       = "sudo tail -f /var/log/cloud-init-output.log"
}

output "nip_io_hostname" {
  description = "A free hostname that resolves to the instance and that Let's Encrypt will issue for, if you would rather not buy a domain."
  value       = var.use_elastic_ip ? "${replace(aws_eip.gpu[0].public_ip, ".", "-")}.nip.io" : null
}
