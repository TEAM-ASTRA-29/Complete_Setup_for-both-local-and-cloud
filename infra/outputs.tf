output "instance_public_ip" {
  value = aws_instance.benchmark.public_ip
}

output "ssh_command" {
  value = "ssh -i <path-to-your-key.pem> ubuntu@${aws_instance.benchmark.public_ip}"
}

output "streamlit_url" {
  value = "http://${aws_instance.benchmark.public_ip}:8501"
}

output "instance_type_used" {
  value = var.instance_type
}
