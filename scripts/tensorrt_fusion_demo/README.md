# Bias + Activation fusion on Brev (TensorRT Layer Fusion demo)

**Interview / DevRel soundbite:** TensorRT’s **Layer Fusion** merges consecutive
ops such as **GEMM/Conv + Bias + ReLU/GELU** into one kernel (often as a GEMM
epilogue). You normally *don’t* hand-write that fusion for TRT — the **builder**
does it. Writing custom CUDA from Python (CuPy / Numba) is a *different* path
where *you* author the fused kernel.

This folder is a small, Brev-friendly demo of all three angles.

| Path | What it shows |
|------|----------------|
| **(A)** CuPy / Numba | Python → CUDA: *you* fuse `y = relu(x + bias)` |
| **(B)** TensorRT Layer Fusion | Builder graph opt merges Bias+Act (not hand-written) |
| **(C)** ONNX → TRT | Export `Linear+ReLU`, build engine, inspect fused layers |

If TensorRT isn’t installed, the script still exports ONNX and falls back to
`torch.compile` latency (Inductor fusion — analogous story, different compiler).

---

## 1 · Answer in 30 seconds

**“Through Brev, can I do Python → CUDA for TensorRT fusion of activation and bias?”**

- **Yes, on a Brev GPU** you can run the full stack (PyTorch → ONNX → TensorRT).
- **Clarify the two meanings of “fusion”:**
  1. **Custom CUDA from Python** — CuPy `RawKernel` / `numba.cuda` for a fused
     bias+ReLU. Useful for teaching CUDA; *not* how TensorRT usually works.
  2. **TensorRT Layer Fusion** — automatic: export a network that *contains*
     separate Bias + Activation nodes; the TensorRT builder fuses them when
     building the engine. Evidence: engine inspector / `trtexec --dumpLayerInfo`
     shows a single fused / `PWN(...)` style layer instead of separate Gemm+Relu.

---

## 2 · Run on NVIDIA Brev (recommended)

### Launch GPU instance

```bash
# From your laptop
brew install brevdev/homebrew-brev/brev || pip install brev
brev login
brev create trt-fusion --gpu "nebius.l40sx1.pcie"   # or any CUDA GPU type
brev shell trt-fusion
```

Same pattern as `reg-agents/brev/README.md` (L40S / A100 / H100 all fine).

### Copy this demo onto the VM

```bash
# Option A: clone study pack if you keep it in git; or scp this folder:
#   scp -r demos/tensorrt_layer_fusion trt-fusion:~/
git clone <your-NVIDIA-study-remote> ~/NVIDIA-study   # if applicable
cd ~/NVIDIA-study/demos/tensorrt_layer_fusion
```

Or from the `reg-agents` checkout on Brev:

```bash
cd ~/reg-agents/scripts/tensorrt_fusion_demo
```

### Easiest TensorRT path: NGC container

Brev images already have Docker + NVIDIA Container Toolkit:

```bash
cd ~/NVIDIA-study/demos/tensorrt_layer_fusion   # or the reg-agents copy

docker run --gpus all -it --rm \
  -v "$PWD":/demo -w /demo \
  nvcr.io/nvidia/tensorrt:24.08-py3 bash

# inside the container:
pip install --quiet torch onnx
python demo_bias_act_fusion.py

# optional CLI evidence of fusion:
trtexec --onnx=artifacts/linear_bias_relu.onnx --fp16 --verbose \
        --dumpLayerInfo --exportLayerInfo=artifacts/layer_info.json
grep -E 'PWN|Fused|Relu|Gemm' artifacts/layer_info.json | head
```

What you should see:

1. **ONNX** lists separate nodes (`Gemm` then `Relu`, or `MatMul`/`Add`/`Relu`).
2. **Engine inspector / layer_info** collapses them (fused / `PWN(Gemm + Relu)`-style).
3. Optional: TRT median latency vs `torch.compile` / eager.

### Bare-metal on the VM (no Docker)

Only if the image already has a matching CUDA + TensorRT wheel:

```bash
pip install torch onnx
# TensorRT: prefer NGC container; pip wheels are version-sensitive.
python demo_bias_act_fusion.py
```

Without TensorRT the script still runs the **torch.compile** fallback and prints
the CuPy/Numba skip messages — good enough to rehearse the talking points.

### Optional: custom CUDA kernel (path A)

```bash
# Match CUDA major version on the VM (example: CUDA 12.x)
pip install cupy-cuda12x
# or: pip install numba
python demo_bias_act_fusion.py
```

---

## 3 · Local machine (no GPU)

```bash
pip install torch onnx
python demo_bias_act_fusion.py --skip-custom-cuda
```

Expect: ONNX export + conceptual prints + CPU `torch.compile` attempt.
No real fusion evidence until you run on Brev / NGC.

---

## 4 · Exam / interview talking points (NCA-GENL)

From `NCA-GENL/06_nvidia_stack_nemo_triton_tensorrt_rapids_cuda.md`:

| Optimization | Idea |
|---|---|
| **Layer fusion** | Merge consecutive ops (e.g. conv+ReLU) → less memory traffic |
| Kernel auto-tuning | Pick fast implementations per GPU |
| Precision (FP16/INT8/…) | Quantized engines |

**Not** TensorRT’s job: inventing residual connections, data augmentation, etc.

**One-liner:** *“Bias and activation ride the matmul epilogue — one memory pass
instead of three.”*

---

## 5 · Files

| File | Role |
|------|------|
| `demo_bias_act_fusion.py` | End-to-end demo |
| `requirements.txt` | Minimal pip deps (TRT via NGC preferred) |
| `artifacts/` | Created at runtime (ONNX, engine, meta) |

Mirror in `reg-agents/scripts/tensorrt_fusion_demo/` for demos already on a Brev
`reg-agents` instance.
