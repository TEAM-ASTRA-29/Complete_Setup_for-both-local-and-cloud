# Cloud Benchmark Setup — Step by Step

## Step 0 — Check your local machine's specs (do this first)

For the comparison to mean anything, you need to know what you're actually
comparing. Run one of these on your local machine:

```bash
# Linux
nproc --all          # vCPU count
free -h               # RAM

# macOS
sysctl -n hw.ncpu     # vCPU count
sysctl -n hw.memsize  # RAM (bytes)

# Windows (PowerShell)
(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
```

Then pick the closest matching EC2 instance type in `infra/variables.tf`:

| Your local machine | Closest EC2 type |
|---|---|
| 2 vCPU / 4 GB  | `t3.medium` |
| 2 vCPU / 8 GB  | `t3.large` |
| 4 vCPU / 8 GB  | `t3.xlarge` or `c5.xlarge` |
| 4 vCPU / 16 GB | `t3.xlarge` |
| 8 vCPU / 16 GB | `c5.2xlarge` |

Note it won't be a perfect match (EC2 vCPUs are shared/virtualized differently
than a physical laptop core) — that's expected and fine to state as a
limitation in your report. The point is getting close, not identical.

If your project's whole premise is comparing *multiple* instance sizes
(which your README suggests), keep one run at the **matched** instance type
as your fair baseline, then add `t3.large` / `c5.xlarge` / etc. as additional
comparison points on top of that baseline.

---

## Step 1 — Prerequisites

- AWS account (you have this)
- [Terraform](https://developer.hashicorp.com/terraform/install) installed locally
- AWS CLI configured: `aws configure` (needs your access key + secret key)
- An EC2 key pair for SSH — create one if you don't have it:
  ```bash
  aws ec2 create-key-pair --key-name cropml-key --query 'KeyMaterial' --output text > cropml-key.pem
  chmod 400 cropml-key.pem
  ```
- Your current public IP: `curl https://checkip.amazonaws.com`

---

## Step 2 — Deploy the infrastructure

```bash
cd infra
terraform init
terraform validate      # do this before apply — sanity-checks syntax against the real AWS provider
```

Create `infra/terraform.tfvars`:
```hcl
instance_type  = "t3.medium"        # set to your matched type from Step 0
key_pair_name  = "cropml-key"
my_ip_cidr     = "YOUR.IP.HERE/32"  # from checkip.amazonaws.com above
```

```bash
terraform plan     # review what will be created — 1 VPC, 1 subnet, 1 IGW, 1 route table, 1 SG, 1 EC2 instance
terraform apply    # type 'yes' to confirm
```

Note the outputs — `instance_public_ip`, `ssh_command`, `streamlit_url`.

---

## Step 3 — Get the app onto the instance

```bash
# from your project root, on your local machine
scp -i cropml-key.pem -r cropml_v4 ubuntu@<INSTANCE_IP>:~/cropml_v4
```

Wait ~60 seconds after `terraform apply` finishes before SSHing in — the
`user_data` script is still installing Docker in the background.

---

## Step 4 — Build the image identically on both sides

**On your local machine:**
```bash
cd cropml_v4
docker build -t cropml:bench .
```

**On the EC2 instance:**
```bash
ssh -i cropml-key.pem ubuntu@<INSTANCE_IP>
cd cropml_v4
docker build -t cropml:bench .
```

Same Dockerfile, same base image, same dependency versions on both — this is
what makes the comparison fair. No manual pip installs on either side.

---

## Step 5 — Run the identical benchmark on both sides

Put your dataset at `cropml_v4/data/<name>.csv` on both machines (same file).

**Local:**
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/benchmark_results:/app/benchmark_results \
  --entrypoint python cropml:bench benchmark_runner.py \
  --data data/<name>.csv --target <target_col> --task Regression \
  --models rf,dt,lr,xgb,gbm --env local --label "my-laptop" --scale --repeats 3
```

**On EC2 (same command, just change `--env` and `--label`):**
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/benchmark_results:/app/benchmark_results \
  --entrypoint python cropml:bench benchmark_runner.py \
  --data data/<name>.csv --target <target_col> --task Regression \
  --models rf,dt,lr,xgb,gbm --env cloud --label "t3.medium" --scale --repeats 3
```

`--repeats 3` runs each model 3 times so you can report mean ± variance
instead of a single noisy number — worth doing for a final-year report.

---

## Step 6 — Pull cloud results back and compare

```bash
scp -i cropml-key.pem ubuntu@<INSTANCE_IP>:~/cropml_v4/benchmark_results/cloud_*.json ./cropml_v4/benchmark_results/
```

Now `benchmark_results/` on your laptop has both `local_*.json` and
`cloud_*.json` — real measured numbers, not the old simulated formula.

---

## Step 7 — Tear down when done (avoid ongoing charges)

```bash
cd infra
terraform destroy
```

Run this every time you're done with a session — EC2 billing is per-hour
while running.
