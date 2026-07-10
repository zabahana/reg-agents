# Deploying reg-agents to GKE (with a GPU node pool)

This runs the **real NVIDIA stack** — NIM (LLM), NeMo Retriever (embeddings),
and Triton (fraud model) — on GPUs, with the MCP servers and A2A agents on CPU.

## 0. Prereqs
- `gcloud`, `kubectl`, Docker
- An NGC API key (https://org.ngc.nvidia.com) for `nvcr.io` images + NIM runtime
- A GCP project with the Kubernetes Engine + Artifact Registry APIs enabled

## 1. Create the cluster + a GPU node pool
```bash
PROJECT=$(gcloud config get-value project)
REGION=us-central1
ZONE=us-central1-a

gcloud container clusters create reg-agents \
  --zone "$ZONE" --num-nodes 2 --machine-type e2-standard-4

# GPU node pool (L4 is cost-effective and enough for 8B NIM + retriever + Triton;
# bump count/type if you co-locate all three, or split into 2-3 GPU nodes).
gcloud container node-pools create gpu-pool \
  --cluster reg-agents --zone "$ZONE" \
  --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
  --num-nodes 3 --node-labels=cloud.google.com/gke-accelerator=nvidia-l4

# Install NVIDIA drivers (GKE managed):
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

gcloud container clusters get-credentials reg-agents --zone "$ZONE"
```

## 2. Build & push the app image
```bash
REPO=us-docker.pkg.dev/$PROJECT/reg-agents
gcloud artifacts repositories create reg-agents --repository-format=docker --location=us
gcloud auth configure-docker us-docker.pkg.dev

docker build -t $REPO/app:latest .
docker push $REPO/app:latest

# Substitute the image into the manifests:
sed -i '' "s#REPLACE_IMAGE#$REPO/app:latest#g" k8s/20-mcp-servers.yaml k8s/30-agents.yaml k8s/40-ui-ingress.yaml
```

## 3. Secrets (NGC for NVIDIA images/runtime)
```bash
kubectl apply -f k8s/00-namespace.yaml

kubectl -n reg-agents create secret docker-registry ngc-secret \
  --docker-server=nvcr.io --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY"

kubectl -n reg-agents create secret generic ngc-api \
  --from-literal=NGC_API_KEY="$NGC_API_KEY"
```

## 4. Deploy
```bash
kubectl apply -f k8s/01-config.yaml
kubectl apply -f k8s/10-nvidia-nim.yaml   # GPU tier (NIM, NeMo Retriever, Triton)
kubectl apply -f k8s/20-mcp-servers.yaml
kubectl apply -f k8s/30-agents.yaml
kubectl apply -f k8s/40-ui-ingress.yaml

kubectl -n reg-agents get pods -w
```

NIM pods take a few minutes to pull the model and become ready
(`/v1/health/ready`). Then grab the UI external IP:
```bash
kubectl -n reg-agents get svc ui
```

## 5. Triton model repository
`triton` expects a model repo at `/models` containing `fraud_xgb_gnn/`
(`config.pbtxt` + the exported GNN+XGBoost model, e.g. FIL backend for XGBoost).
Back it with a GCS bucket via the gcsfuse CSI driver or a PVC. Until then the
fraud MCP server automatically uses its local heuristic and reports
`backend: heuristic-local`.

## Cost note
GPUs are the expensive part. Scale the GPU pool to 0 when idle:
```bash
gcloud container clusters resize reg-agents --node-pool gpu-pool --num-nodes 0 --zone "$ZONE"
```
