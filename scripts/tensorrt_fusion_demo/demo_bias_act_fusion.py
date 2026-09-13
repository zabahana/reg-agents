#!/usr/bin/env python3
"""
Bias + Activation fusion demo (interview / DevRel talking point).

Three paths, clearly separated:

  (A) Custom CUDA from Python  — optional CuPy / Numba fused bias+ReLU kernel
  (B) TensorRT Layer Fusion    — builder graph opt merges Gemm/Conv + Bias + Act
                                  (you usually do NOT hand-write the fusion)
  (C) ONNX → TensorRT path     — export Linear+ReLU, build engine, inspect layers

Fallback when TensorRT is missing: torch.compile (Inductor) kernel fusion +
latency compare vs eager PyTorch.

Run on an NVIDIA Brev GPU (see README.md). Safe to run CPU-only for ONNX export
and the conceptual prints; GPU needed for meaningful latency / fusion evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms"


def _has_cuda_torch() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model: trivial Linear (Gemm+Bias) + ReLU  — classic TensorRT fusion pattern
# ---------------------------------------------------------------------------


def build_torch_model(in_features: int = 1024, out_features: int = 1024):
    import torch
    import torch.nn as nn

    class LinearBiasReLU(nn.Module):
        """Explicit Linear (weight+bias) then ReLU — TRT fuses these."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(in_features, out_features, bias=True)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):  # noqa: ANN001
            return self.act(self.fc(x))

    model = LinearBiasReLU().eval()
    return model


def export_onnx(model, onnx_path: Path, batch: int, in_features: int) -> Path:
    import torch

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(batch, in_features)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote ONNX: {onnx_path} ({onnx_path.stat().st_size} bytes)")
    _print_onnx_graph_summary(onnx_path)
    return onnx_path


def _print_onnx_graph_summary(onnx_path: Path) -> None:
    try:
        import onnx

        graph = onnx.load(str(onnx_path)).graph
        ops = [n.op_type for n in graph.node]
        print(f"ONNX nodes ({len(ops)}): {ops}")
        print(
            "  → Pre-fusion: expect Gemm (or MatMul+Add) then Relu as separate ops."
        )
    except ImportError:
        print("  (install onnx to print graph node list: pip install onnx)")


# ---------------------------------------------------------------------------
# (A) Custom CUDA from Python — optional fused bias+ReLU
# ---------------------------------------------------------------------------


def demo_custom_cuda_python(n: int = 1 << 20) -> None:
    _banner("(A) Custom CUDA from Python (CuPy / Numba) — hand-written fusion")
    print(
        "Yes: you can write CUDA kernels from Python (CuPy RawKernel, numba.cuda,\n"
        "pycuda). That is *authoring* a fused bias+activation kernel yourself.\n"
        "TensorRT Layer Fusion is different: the *builder* merges ops for you."
    )

    x = np.random.randn(n).astype(np.float32)
    bias = np.random.randn(n).astype(np.float32)

    # CuPy path
    try:
        import cupy as cp

        kernel = cp.RawKernel(
            r"""
            extern "C" __global__
            void bias_relu(const float* x, const float* b, float* y, int n) {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                if (i < n) {
                    float v = x[i] + b[i];
                    y[i] = v > 0.f ? v : 0.f;
                }
            }
            """,
            "bias_relu",
        )
        dx, db = cp.asarray(x), cp.asarray(bias)
        dy = cp.empty_like(dx)
        threads = 256
        blocks = (n + threads - 1) // threads
        # warmup
        kernel((blocks,), (threads,), (dx, db, dy, n))
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            kernel((blocks,), (threads,), (dx, db, dy, n))
        cp.cuda.Stream.null.synchronize()
        dt = (time.perf_counter() - t0) / 50
        y_ref = np.maximum(x + bias, 0.0)
        err = float(np.max(np.abs(cp.asnumpy(dy) - y_ref)))
        print(f"CuPy RawKernel fused bias+ReLU: {_ms(dt)}/iter  max_abs_err={err:.2e}")
        return
    except Exception as e:
        print(f"CuPy path skipped: {e}")

    # Numba path
    try:
        from numba import cuda

        @cuda.jit
        def bias_relu_kernel(xv, bv, yv):
            i = cuda.grid(1)
            if i < xv.size:
                v = xv[i] + bv[i]
                yv[i] = v if v > 0.0 else 0.0

        dx = cuda.to_device(x)
        db = cuda.to_device(bias)
        dy = cuda.device_array_like(dx)
        threads = 256
        blocks = (n + threads - 1) // threads
        bias_relu_kernel[blocks, threads](dx, db, dy)
        cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            bias_relu_kernel[blocks, threads](dx, db, dy)
        cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50
        y_host = dy.copy_to_host()
        err = float(np.max(np.abs(y_host - np.maximum(x + bias, 0.0))))
        print(f"Numba CUDA fused bias+ReLU: {_ms(dt)}/iter  max_abs_err={err:.2e}")
        return
    except Exception as e:
        print(f"Numba path skipped: {e}")

    print(
        "No CuPy/Numba CUDA available — conceptual only on this machine.\n"
        "On Brev: pip install cupy-cuda12x  (match the VM CUDA) or numba."
    )


# ---------------------------------------------------------------------------
# (B)/(C) TensorRT Python API — build engine and report fusion
# ---------------------------------------------------------------------------


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    workspace_mb: int = 256,
) -> dict[str, Any]:
    """Build a TensorRT engine from ONNX; return layer / timing metadata."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    print(f"Network layers BEFORE builder optimization: {network.num_layers}")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"  [{i}] {layer.name!r}  type={layer.type}")

    config = builder.create_builder_config()
    # TensorRT API differs slightly across versions
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    else:
        config.max_workspace_size = workspace_mb << 20  # type: ignore[attr-defined]

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    # Capture optimization profile for dynamic batch
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    # shapes: min / opt / max
    c = int(inp.shape[1]) if inp.shape[1] > 0 else 1024
    profile.set_shape(inp.name, (1, c), (8, c), (64, c))
    config.add_optimization_profile(profile)

    print("Building engine (this is where Layer Fusion runs)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Wrote engine: {engine_path} ({engine_path.stat().st_size} bytes)")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)

    meta: dict[str, Any] = {
        "num_io_tensors": getattr(engine, "num_io_tensors", None),
        "layers_inspected": [],
        "fusion_heuristic": None,
    }

    # Inspector API (TRT 8.6+) — best evidence of fusion
    try:
        inspector = engine.create_engine_inspector()
        # JSON layer info often shows fused names like "PWN(Gemm_... + Relu_...)"
        info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        meta["inspector_json"] = info
        layers = json.loads(info) if info.strip().startswith("[") or info.strip().startswith("{") else info
        # Print a short summary
        text = info if isinstance(info, str) else json.dumps(layers)
        print("\n--- Engine inspector (look for fused Gemm/Relu / PWN / Fused*) ---")
        # Keep output readable
        for line in text.splitlines()[:80]:
            print(line)
        if len(text.splitlines()) > 80:
            print(f"... ({len(text.splitlines()) - 80} more lines)")
        fused_hints = [
            tok
            for tok in (
                "PWN",
                "Fused",
                "Gemm_Relu",
                "Conv_Relu",
                "ReLU",
                "Relu",
                "Activation",
            )
            if tok in text
        ]
        meta["fusion_heuristic"] = fused_hints
        print(f"\nFusion-related tokens found in inspector output: {fused_hints or '(none parsed)'}")
        print(
            "Interview line: 'Builder fused Bias+ReLU into the GEMM epilogue —\n"
            "  fewer kernel launches, less DRAM traffic than running them separate.'"
        )
    except Exception as e:
        print(f"Engine inspector unavailable ({e}); trying layer names via bindings...")
        # Older fallback: list IO tensor names only
        n_io = getattr(engine, "num_io_tensors", 0) or 0
        for i in range(n_io):
            name = engine.get_tensor_name(i)
            meta["layers_inspected"].append(name)
            print(f"  IO tensor: {name}")

    return meta


def benchmark_trt(engine_path: Path, batch: int, in_features: int, iters: int = 100) -> float:
    """Return median latency seconds for TRT inference (GPU)."""
    import tensorrt as trt

    try:
        import cupy as cp

        use_cupy = True
    except ImportError:
        use_cupy = False
        try:
            from cuda import cudart  # type: ignore  # noqa: F401
        except Exception:
            pass

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    # Allocate with torch if possible (simplest portable path)
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available — skip TRT benchmark")
        return float("nan")

    x = torch.randn(batch, in_features, device="cuda", dtype=torch.float32)
    y = torch.empty(batch, in_features, device="cuda", dtype=torch.float32)

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    # Heuristic: find input/output by mode
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inp_name = name
        else:
            out_name = name

    context.set_input_shape(inp_name, tuple(x.shape))
    context.set_tensor_address(inp_name, x.data_ptr())
    context.set_tensor_address(out_name, y.data_ptr())

    # Warmup
    for _ in range(20):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    med = times[len(times) // 2]
    print(f"TensorRT median latency (batch={batch}): {_ms(med)}")
    return med


# ---------------------------------------------------------------------------
# Fallback: torch.compile fusion evidence
# ---------------------------------------------------------------------------


def demo_torch_compile_fallback(
    model,
    batch: int,
    in_features: int,
    iters: int = 100,
) -> None:
    _banner("Fallback: torch.compile (Inductor) — fusion without TensorRT")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.randn(batch, in_features, device=device)

    def bench(m, tag: str) -> float:
        # warmup
        with torch.inference_mode():
            for _ in range(20):
                _ = m(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(iters):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = m(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        print(f"  {tag}: {_ms(med)} median ({device})")
        return med

    eager = bench(model, "eager Linear+ReLU")
    compiled_med = float("nan")
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        compiled_med = bench(compiled, "torch.compile")
        if eager > 0 and compiled_med == compiled_med:
            speedup = eager / compiled_med
            print(f"  speedup vs eager: {speedup:.2f}x")
            print(
                "  Inductor fuses pointwise ops into fewer Triton/CUDA kernels\n"
                "  (analogous *story* to TRT layer fusion; different compiler)."
            )
    except Exception as e:
        print(f"  torch.compile unavailable: {e}")

    # Optional: show that separate Add+ReLU vs fused is the point
    print(
        "\nTalking point: without fusion you pay for (1) GEMM write to DRAM,\n"
        "  (2) bias+ReLU read/write. With fusion, bias+ReLU ride the GEMM\n"
        "  epilogue — one pass over the output tile."
    )
    return


def run_trtexec_hint(onnx_path: Path) -> None:
    print(
        "\nOptional CLI evidence on Brev (NGC TensorRT image has trtexec):\n"
        f"  trtexec --onnx={onnx_path} --fp16 --verbose --dumpLayerInfo \\\n"
        "          --exportLayerInfo=layer_info.json\n"
        "Then grep layer_info.json / verbose log for 'PWN', 'Fused', or a single\n"
        "layer spanning Gemm+Relu — that is Layer Fusion."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Bias+Activation fusion demo (TRT / CUDA / torch.compile)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--features", type=int, default=1024)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--skip-custom-cuda", action="store_true")
    p.add_argument("--skip-trt", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    _banner("TensorRT / CUDA bias+activation fusion — demo")
    print(f"Python: {sys.version.split()[0]}  cwd={os.getcwd()}")
    print(f"CUDA torch available: {_has_cuda_torch()}")
    print(f"TensorRT Python package: {_trt_available()}")

    print(
        """
Quick map (interview answer):
  (a) Python→CUDA kernels (CuPy/Numba/pycuda): YOU write fused bias+act.
  (b) TensorRT Layer Fusion: BUILDER merges Gemm/Conv + Bias + Act automatically.
  (c) Demo path: PyTorch → ONNX → TensorRT engine → inspector / trtexec layer info.
"""
    )

    if not args.skip_custom_cuda:
        demo_custom_cuda_python()

    _banner("(C) PyTorch → ONNX (unfused graph)")
    try:
        import torch
    except ImportError:
        print("PyTorch required: pip install torch")
        return 1

    model = build_torch_model(args.features, args.features)
    onnx_path = args.out / "linear_bias_relu.onnx"
    export_onnx(model, onnx_path, args.batch, args.features)

    trt_ran = False
    if not args.skip_trt and _trt_available():
        _banner("(B)/(C) TensorRT build — Layer Fusion happens here")
        engine_path = args.out / "linear_bias_relu.engine"
        try:
            meta = build_trt_engine(
                onnx_path, engine_path, fp16=not args.no_fp16
            )
            (args.out / "trt_meta.json").write_text(
                json.dumps(
                    {k: v for k, v in meta.items() if k != "inspector_json"},
                    indent=2,
                    default=str,
                )
            )
            if _has_cuda_torch():
                benchmark_trt(engine_path, args.batch, args.features, args.iters)
            trt_ran = True
        except Exception as e:
            print(f"TensorRT build/bench failed: {e}")
            print("Falling through to torch.compile demo.")
    else:
        _banner("TensorRT not installed — skipping engine build")
        print(
            "On Brev, easiest path is the NGC container:\n"
            "  docker run --gpus all -it --rm -v \"$PWD\":/demo \\\n"
            "    nvcr.io/nvidia/tensorrt:24.08-py3\n"
            "  cd /demo && pip install torch onnx && python demo_bias_act_fusion.py"
        )

    run_trtexec_hint(onnx_path)

    if not trt_ran:
        demo_torch_compile_fallback(model, args.batch, args.features, args.iters)
    else:
        # Still show compile as a side-by-side story if CUDA present
        if _has_cuda_torch():
            demo_torch_compile_fallback(model, args.batch, args.features, args.iters)

    _banner("Done")
    print(f"Artifacts under: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
