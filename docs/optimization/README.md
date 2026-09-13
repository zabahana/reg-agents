# Inference optimization evidence

This directory keeps optimization work separate from the model-validation
record. A serving claim requires a committed result JSON, the engine/profile
configuration, and a quality comparison—not an assumption that a deployment
manifest makes a model faster.

## 1. Precision benchmark: FP16 versus INT8

TensorRT-LLM/NIM selects precision when its engine is built or its compatible
NIM profile is deployed. The benchmark script **records** that chosen profile;
it intentionally does not pretend an API request can convert a hosted model to
INT8.

Run against separately deployed FP16 and INT8 endpoints:

```bash
export NIM_API_KEY='nvapi-...'

python scripts/benchmark_nim_serving.py \
  --base-url http://nim-fp16:8000/v1 --profile fp16-trtllm \
  --engine-settings '{"precision":"fp16","engine":"TensorRT-LLM","kv_cache":"paged"}'

python scripts/benchmark_nim_serving.py \
  --base-url http://nim-int8:8000/v1 --profile int8-trtllm \
  --engine-settings '{"precision":"int8","quantization":"weight-only","engine":"TensorRT-LLM","kv_cache":"paged"}'
```

The output JSON captures p50/p95 time-to-first-token, total latency, and
estimated output tokens/s. Record GPU memory and quality (the same golden
prompt set) alongside those results before selecting a profile. Hosted NIM
does not expose an operator-controlled precision switch, so it is unsuitable
for this A/B comparison.

### Controlled concurrency sweep

Hold the prompt, generation limit, model version, and engine profile fixed.
Run each level after the engine is warm:

```bash
for concurrency in 1 4 8; do
  python scripts/benchmark_nim_serving.py \
    --base-url http://nim-fp16:8000/v1 --profile "fp16-c${concurrency}" \
    --concurrency "$concurrency" --runs 10 --warmup 2 \
    --engine-settings '{"precision":"fp16","engine":"TensorRT-LLM","kv_cache":"paged"}'
done
```

Compare TTFT p95, total-latency p95, output tokens/s, GPU memory, and request
errors. Stop increasing concurrency when p95 grows beyond the agreed service
objective or the server rejects/OOMs requests.

### Quality comparison

The reserved CFPB scoring holdout has weak labels. The script below reports
its agreement separately for the stage-1 gate, end-to-end output, and only
the actual `rag_llm` rows. It labels the result accurately as weak-label
agreement, not human-adjudicated accuracy.

```bash
NIM_BASE_URL=http://nim-fp16:8000/v1 \
  python scripts/evaluate_complaint_quality.py --profile fp16-trtllm --limit 20
NIM_BASE_URL=http://nim-int8:8000/v1 \
  python scripts/evaluate_complaint_quality.py --profile int8-trtllm --limit 20
```

### Self-hosted NIM/TensorRT-LLM on Brev

The hosted NVIDIA catalog is useful for functional testing, but it does not
provide an operator-controlled precision profile. The
`docker-compose.nim-selfhosted.yml` overlay starts Llama 3.1 8B locally on the
GPU and requires an explicit NIM profile so every measurement is attributable.

```bash
# Obtain an NGC API key with access to nvcr.io; do not commit or echo it.
export NGC_API_KEY='...'

# The exact image/GPU determines the available profile IDs and precisions.
docker run --rm --gpus all -e NGC_API_KEY \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest list-model-profiles

# Choose a profile whose output explicitly identifies its precision, then start it.
export NIM_MODEL_PROFILE='<fp16-profile-id>'
COMPOSE_PARALLEL_LIMIT=1 docker compose \
  -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.nim-selfhosted.yml --profile monitoring up -d nim-llm

until curl -fsS http://localhost:8003/v1/health/ready >/dev/null; do sleep 10; done
```

Run the same profile at concurrency 1, 4, and 8, then stop the service, choose
the verified INT8 profile ID, and repeat. The app containers do not need to
switch providers for the measurement commands: use the Docker-service endpoint
`http://nim-llm:8000/v1` explicitly.

```bash
docker compose exec -T complaint-mcp python scripts/benchmark_nim_serving.py \
  --base-url http://nim-llm:8000/v1 --profile fp16-c1 --concurrency 1 \
  --runs 10 --warmup 2 \
  --engine-settings '{"precision":"fp16","engine":"TensorRT-LLM","kv_cache":"paged"}'

docker compose exec -T complaint-mcp sh -lc \
  'NIM_BASE_URL=http://nim-llm:8000/v1 NIM_MODEL=meta/llama-3.1-8b-instruct \
   python scripts/evaluate_complaint_quality.py --profile fp16-trtllm --limit 20'
```

## 2. KV cache

TensorRT-LLM manages a paged KV cache during generation. The self-hosted
manifest bounds its capacity through `NIM_MAX_MODEL_LEN=4096` and
`NIM_MAX_BATCH_SIZE=8`. These are capacity guardrails—not proof of cache
benefit. Verify the values in NIM startup logs for the selected profile, then
benchmark short and long prompts at concurrency 1, 4, and 8.

Do not set an undocumented vendor-specific “KV cache percentage” environment
variable. Its accepted name and behavior can differ by NIM image/profile.

## 3. DistilBERT pruning challenger

```bash
# Requires the DPO policy artifact plus optional torch/transformers packages.
python scripts/prune_distilbert.py --device cuda --amount 0.30
```

The script applies global L1 magnitude pruning to the DPO DistilBERT
challenger and evaluates it on the held-out stage-1 fold. It reports
ROC-AUC/F1/precision/recall, linear-layer sparsity, and inference latency.

Dense zero weights are not automatically smaller or faster. Promote pruning
only after exporting to a sparse or structured-runtime format and comparing
quality, p95 latency, throughput, GPU memory, and cost to the unpruned
baseline. The logistic-regression gate remains the production stage-1 path.

## Acceptance gates

| Change | Required evidence before promotion |
|---|---|
| FP16/INT8 engine | Golden-set quality non-regression, p95 latency/throughput, GPU memory, engine version |
| KV-cache sizing | Startup configuration, workload-concurrency test, OOM/rejection rate, TTFT |
| Pruned challenger | Held-out metrics, structured/sparse export, latency and memory versus baseline |
