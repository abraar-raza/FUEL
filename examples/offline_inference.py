from vllm import LLM, SamplingParams
from vllm.engine.metrics import LoggingStatLogger

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    # "The stock price of Nvidia will",
    # "The weather in New York is",
    # "The meaning of life is",
    # "The best movie of all time is",
    # "The best programming language is",
    # "The best book of all time is",
    # "The best restaurant in the world is",
    # "The best place to visit in the world is",
    # "The best time to visit Paris is",
    # "The best time to visit New York is",
    # "The best time to visit London is",
    # "The best time to visit San Francisco is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)
logger = LoggingStatLogger(local_interval=0.25)

# Create an LLM.
llm = LLM(model="/path/to/model",disable_log_stats=False, tensor_parallel_size=1)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    print(output, len(output.prompt_token_ids), len(output.outputs[0].token_ids))
    print(output.metrics)
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")