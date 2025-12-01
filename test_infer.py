from vllm import LLM, SamplingParams
import numpy as np

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/data01/huawei-2025/weight/Qwen3_30B/Qwen3_30B", enforce_eager=True,
          distributed_executor_backend="mp", tensor_parallel_size=8, enable_return_routed_experts=True)

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
