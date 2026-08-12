"""Triton Python backend for the CMPL-REG-24 stage-1 gate.

Loads ``artifacts.joblib`` produced by
``scripts/export_complaint_triton_model.py``:

    {vectorizer, model, threshold, champion}
"""

from __future__ import annotations

import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore


class TritonPythonModel:
    def initialize(self, args):
        model_dir = args["model_repository"]
        version = args["model_version"]
        art_path = os.path.join(model_dir, version, "artifacts.joblib")
        meta_path = os.path.join(model_dir, version, "meta.json")
        try:
            import joblib
            self.artifacts = joblib.load(art_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"complaint_stage1: failed to load {art_path}: {exc}"
            ) from exc
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                self.meta = json.load(fh)

    def execute(self, requests):
        responses = []
        vec = self.artifacts["vectorizer"]
        model = self.artifacts["model"]
        thr = float(self.artifacts["threshold"])
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "NARRATIVE")
            # Triton STRING tensors arrive as numpy object/bytes.
            raw = inp.as_numpy().reshape(-1)
            texts = []
            for item in raw:
                if isinstance(item, bytes):
                    texts.append(item.decode("utf-8"))
                else:
                    texts.append(str(item))
            xt = vec.transform(texts)
            proba = model.predict_proba(xt)[:, 1].astype(np.float32)
            thr_arr = np.full_like(proba, thr, dtype=np.float32)
            out_prob = pb_utils.Tensor("PROBABILITY", proba.reshape(-1, 1))
            out_thr = pb_utils.Tensor("THRESHOLD", thr_arr.reshape(-1, 1))
            responses.append(pb_utils.InferenceResponse([out_prob, out_thr]))
        return responses

    def finalize(self):
        self.artifacts = None
