"""Shared helpers for KernelBench backends."""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from kernelgym.toolkit.kernelbench.loading import graceful_eval_cleanup
from kernelgym.backend.base import Backend


class KernelBenchBackendBase(Backend):
    name = "kernelbench.base"

    @staticmethod
    def _normalize_device(device: Any | None) -> torch.device:
        if device is None:
            return torch.device("cuda:0")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    @staticmethod
    def _maybe_set_cuda_device(device: torch.device) -> None:
        if device.type != "cuda":
            return
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass

    @staticmethod
    def _maybe_set_triton_env(device: torch.device) -> None:
        if device.type != "cuda":
            return
        device_index = device.index
        if device_index is None:
            try:
                device_index = torch.cuda.current_device()
            except Exception:
                return
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

    @staticmethod
    def _normalize_init_inputs(init_inputs: Any) -> Any:
        if (
            isinstance(init_inputs, list)
            and len(init_inputs) > 1
            and hasattr(init_inputs[0], "__len__")
            and not isinstance(init_inputs[0], (str, torch.Tensor))
            and len(init_inputs[0]) == 0
        ):
            return init_inputs[1]
        return init_inputs

    @staticmethod
    def _move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return type(value)(
                KernelBenchBackendBase._move_to_device(v, device) for v in value
            )
        if isinstance(value, dict):
            return {
                k: KernelBenchBackendBase._move_to_device(v, device)
                for k, v in value.items()
            }
        return value

    def create_model(self, handle: Any, init_inputs: Any, **kwargs: Any) -> Any:
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("KernelBenchBackend.create_model expects a handle from load()")

        device = kwargs.get("device") or handle.get("device") or self._normalize_device(None)
        if not isinstance(device, torch.device):
            device = self._normalize_device(device)
        self._maybe_set_cuda_device(device)
        if handle.get("backend") == "triton":
            self._maybe_set_triton_env(device)
        no_grad = kwargs.get("no_grad", True)
        synchronize = kwargs.get("synchronize", False)

        model_cls = handle["model_cls"]
        init_inputs = self._normalize_init_inputs(init_inputs)
        init_inputs = self._move_to_device(init_inputs, device)

        if no_grad:
            with torch.no_grad():
                if isinstance(init_inputs, dict):
                    model = model_cls(**init_inputs)
                else:
                    model = model_cls(*init_inputs)
        else:
            if isinstance(init_inputs, dict):
                model = model_cls(**init_inputs)
            else:
                model = model_cls(*init_inputs)

        if hasattr(model, "to"):
            model = model.to(device)

        if synchronize and device.type == "cuda":
            torch.cuda.synchronize(device=device)

        return model

    def run(self, handle: Any, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(handle, dict) or "model_cls" not in handle:
            raise ValueError("KernelBenchBackend.run expects a handle from load()")

        device = kwargs.get("device") or handle.get("device") or self._normalize_device(None)
        if not isinstance(device, torch.device):
            device = self._normalize_device(device)
        self._maybe_set_cuda_device(device)
        if handle.get("backend") == "triton":
            self._maybe_set_triton_env(device)
        no_grad = kwargs.get("no_grad", True)
        synchronize = kwargs.get("synchronize", True)

        init_inputs = inputs.get("init_inputs", inputs.get("inputs", []))
        run_inputs = inputs.get("inputs", init_inputs)
        run_inputs = self._move_to_device(run_inputs, device)

        model = self.create_model(
            handle,
            init_inputs,
            device=device,
            no_grad=no_grad,
            synchronize=synchronize,
        )

        if no_grad:
            with torch.no_grad():
                output = (
                    model(**run_inputs)
                    if isinstance(run_inputs, dict)
                    else model(*run_inputs)
                )
        else:
            output = model(**run_inputs) if isinstance(run_inputs, dict) else model(*run_inputs)

        if synchronize and device.type == "cuda":
            torch.cuda.synchronize(device=device)

        return {"output": output}

    def cleanup(self, handle: Any, **kwargs: Any) -> None:
        if not isinstance(handle, dict):
            return
        device = handle.get("device")
        context = handle.get("context", {})
        tempfile_handle = handle.get("tempfile_handle")

        if isinstance(device, torch.device) and device.type == "cuda":
            try:
                graceful_eval_cleanup(context, device, tempfile_handle)
                return
            except Exception:
                pass

        if tempfile_handle is not None:
            try:
                tempfile_handle.close()
            except Exception:
                pass
