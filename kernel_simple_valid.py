import httpx
import asyncio
from uuid import uuid4


async def evaluate_kernel_simple():
    task_id = f"my-kernel-task-{uuid4().hex[:8]}"
    kernel_code = '''
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
'''
    cases_code = '''
import torch

def get_init_inputs():
    return []

def get_cases():
    x = torch.randn(1024, device="cuda")
    y = torch.randn(1024, device="cuda")
    expected = x + y
    return [{"inputs": [x, y], "outputs": expected}]
'''

    timeout = httpx.Timeout(300.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "http://192.168.31.68:8001/workflow/submit",
            json={
                "workflow": "kernel_simple",
                "task_id": task_id,
                "force_refresh": True,
                "payload": {
                    "task_id": task_id,
                    "kernel_code": kernel_code,
                    "cases_code": cases_code,
                    "entry_point": "ModelNew",
                    "backend": "triton",
                    "device": "cuda:0",
                    "run_correctness": True,
                    "run_performance": True,
                    "num_perf_trials": 100,
                }
            }
        )
        response.raise_for_status()
        return response.json()

result = asyncio.run(evaluate_kernel_simple())
print(f"Compiled: {result['result']['compiled']}")
print(f"Correctness: {result['result']['correctness']}")
print(f"Runtime: {result['result']['kernel_runtime']:.4f} ms")
