variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = <<-EOT
    EC2 instance type used for the benchmark.
    IMPORTANT: for a fair local-vs-cloud comparison, this should match
    your local machine's vCPU/RAM as closely as possible. Check your
    local specs first (see infra/README.md), then set this accordingly.
      t3.medium   -> 2 vCPU, 4 GB RAM
      t3.large    -> 2 vCPU, 8 GB RAM
      t3.xlarge   -> 4 vCPU, 16 GB RAM
      c5.xlarge   -> 4 vCPU, 8 GB RAM  (compute-optimized, for a second comparison point)
  EOT
  type    = string
  default = "t3.medium"
}

variable "key_pair_name" {
  description = "Name of an EXISTING EC2 key pair in your AWS account (for SSH access)"
  type        = string
}

variable "my_ip_cidr" {
  description = "Your current public IP in CIDR form, e.g. 203.0.113.5/32. Get it from https://checkip.amazonaws.com. SSH access is restricted to this."
  type        = string
}

variable "project_name" {
  description = "Prefix used to tag/name all resources"
  type        = string
  default     = "cropml-benchmark"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 20
}
