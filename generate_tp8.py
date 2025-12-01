# generate_tp8.py
import torch_npu
import torch.distributed as dist
from vllm import LLM, SamplingParams
import os
import numpy as np

# Initialize torch.distributed (required for external_launcher)
if "RANK" in os.environ:
    dist.init_process_group(backend="hccl")  # 或 "gloo" for CPU debug


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "一加一等于多少",
    ]
    # prompts = [
    #     "<question>Hello, my name is</question>",
    #     "<question>The president of the United States is</question><answer>",
    #     "The capital of France is</question><answer>",
    #     "<question>The future of AI is</question><answer>",
    #     "<question>一加一等于多少？<answer>",
    # ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

    # Only rank 0 runs the LLM engine (driver)
    llm = LLM(
        model="/data01/huawei-2025/gxj/Moonlight-16B-A3B-Instruct/",
        trust_remote_code=True,
        enforce_eager=True,
        distributed_executor_backend="external_launcher",
        tensor_parallel_size=4,
        enable_return_routed_experts=True
        # data_parallel_size=2,
        # Optional: disable Ray, make sure vLLM uses torch.distributed
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        router_info = output.outputs[0].routed_experts
        token_ids = output.outputs[0].token_ids
        prompt_token_ids = output.prompt_token_ids
        np.set_printoptions(threshold=np.inf, linewidth=500)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, prompt token_ids: {prompt_token_ids}"
              f"response token_ids: {token_ids}, router_info len :{len(router_info)}, router_info: {router_info}")

    # Shutdown cleanly
    del llm

    # All ranks (including rank 0) must wait before exit
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()