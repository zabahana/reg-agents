# Deploying reg-agents to GKE

Default profile: **hosted NIM** for the LLM + embeddings (NVIDIA
`build.nvidia.com` catalog — no LLM GPU) and **self-hosted Triton** for the
fraud model. Triton is the only GPU workload, so the GPU pool is a single node.
Prometheus + Grafana (kube-prometheus-stack) give you observability.

```
                      ┌──────────────── GKE cluster ─────────────────┐
  hosted NIM  ◀──────▶│  agents (CPU) ──▶ MCP servers (CPU)          │
  (LLM+embeds)        │        │                                     │
                      │        └──────────────▶ Triton (1× L4 GPU)   │
                      │  kube-prometheus-stack: Prometheus + Grafana │
                      └───────────────────────────────────────────────┘
```

To self-host NIM on GPU instead, see [`optional/nim-selfhosted.yaml`](optional/nim-selfhosted.yaml).

## 0. Prereqs
- `gcloud`, `kubectl`, `helm`, Docker
- A hosted **NIM API key** from https://build.nvidia.com (`nvapi-…`)
- A GCP project with Kubernetes Engine + Artifact Registry APIs enabled
- GPU quota for 1× `nvidia-l4` in your region

## 1. Cluster + a small GPU node pool (Triton only)
```bash
PROJECT=$(gcloud config get-value project)
ZONE=us-central1-a

gcloud container clusters create reg-agents \
  --zone "$ZONE" --num-nodes 2 --machine-type e2-standard-4

# One L4 is enough for the Triton FIL fraud model.
gcloud container node-pools create gpu-pool \
  --cluster reg-agents --zone "$ZONE" \
  --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1 \
  --num-nodes 1 --node-labels=cloud.google.com/gke-accelerator=nvidia-l4

# GKE-managed NVIDIA drivers:
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

## 3. Namespace + secret (hosted NIM key)
```bash
kubectl apply -f k8s/00-namespace.yaml

kubectl -n reg-agents create secret generic reg-agents-secrets \
  --from-literal=NIM_API_KEY="$NIM_API_KEY" \
  --from-literal=OPENAI_API_KEY=""
```
(That replaces the placeholder Secret in `01-config.yaml`; apply the ConfigMap
part with `kubectl apply -f k8s/01-config.yaml` — it's safe, the placeholder
Secret just gets overwritten by the one above.)

## 4. Triton model repository
Generate the fraud model and stage it in a GCS bucket that Triton mounts at
`/models`:
```bash
# Generate fraud_xgb_gnn/{config.pbtxt,1/xgboost.json}
#   (Linux/CI, or macOS with `brew install libomp`, or inside the app image:)
docker run --rm -v "$PWD/triton:/app/triton" $REPO/app:latest \
  python scripts/export_triton_model.py

# Stage in GCS:
gsutil mb -l us gs://$PROJECT-reg-agents-models
gsutil -m cp -r triton/model_repository/* gs://$PROJECT-reg-agents-models/
```
Then mount it via the **gcsfuse CSI driver** (enable with
`--addons GcsFuseCsiDriver` on the cluster) by replacing the Triton
`model-repo` volume in `k8s/10-triton.yaml` with a `csi` volume pointing at the
bucket, or copy the repo into a PVC. Until a model is present, the fraud MCP
server falls back to `backend: heuristic-local` (the app still works).

See [`../triton/README.md`](../triton/README.md) for details.

## 5. Deploy the app
```bash
kubectl apply -f k8s/01-config.yaml
kubectl apply -f k8s/10-triton.yaml        # GPU tier (Triton only)
kubectl apply -f k8s/20-mcp-servers.yaml   # CPU MCP tool servers (+ modeling)
kubectl apply -f k8s/30-agents.yaml        # CPU A2A agents (+ lifecycle)
kubectl apply -f k8s/40-ui-ingress.yaml

kubectl -n reg-agents get pods -w
```
Grab the UI external IP once ready:
```bash
kubectl -n reg-agents get svc ui
```

## 6. Monitoring (Prometheus + Grafana)
Full guide in [`monitoring/README.md`](monitoring/README.md):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts && helm repo update
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f k8s/monitoring/values-kps.yaml

kubectl apply -f k8s/monitoring/servicemonitors.yaml
kubectl apply -f k8s/monitoring/grafana-dashboard.yaml

kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
# http://localhost:3000 → dashboard "reg-agents — agents & Triton" (admin / reg-agents)
```

## Cost note
Only the GPU node costs real money. Scale it to 0 when idle:
```bash
gcloud container clusters resize reg-agents --node-pool gpu-pool --num-nodes 0 --zone "$ZONE"
```
With hosted NIM there's no LLM GPU — you only pay per-token to the NVIDIA
catalog. To self-host NIM instead, apply `optional/nim-selfhosted.yaml`, size
the GPU pool for 3 GPUs, and point `01-config.yaml` back at in-cluster DNS.
