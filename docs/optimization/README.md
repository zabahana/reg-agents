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
