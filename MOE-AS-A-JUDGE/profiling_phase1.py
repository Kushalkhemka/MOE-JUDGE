import os
import json
from collections import defaultdict
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import sys
import logging
import argparse
sys.path.append('../Symbolic-MoE')
# Explicit agent_map from Symbolic-MoE/agent.py
agent_map = {
    "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "Phi": "microsoft/Phi-3.5-mini-instruct",
    "Gemma": "google/gemma-2-9b-it",
    "GLM": "THUDM/glm-4-9b-chat",
    "Exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Granite": "ibm-granite/granite-3.1-8b-instruct",
    "QwenMath": "Qwen/Qwen2.5-Math-7B",
    "QwenCode": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeekMath": "deepseek-ai/deepseek-math-7b-instruct",
    "QwenR1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "LlamaR1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "InternLM": "internlm/internlm3-8b-instruct",
    "Mathstral": "mistralai/Mathstral-7B-v0.1",
    "BioLlama": "ContactDoctor/Bio-Medical-Llama-3-8B",  
    "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama70B": "meta-llama/Llama-3.3-70B-Instruct"
}
from utils import read_json, write_json


"""
Usage Example:
python profiling_phase1.py --model_dir ./saved_models --models Llama,Qwen --batch_size 4 --output results.json --gpus 1

- --model_dir: Path to directory containing model weights (default: ./saved_models)
- --models: Comma-separated list of agent names to profile (default: all in agent_map)
- --batch_size: Batch size for LLM inference (default: 8)
- --output: Output file for profiling results (default: profiling_results.json)
- --gpus: Number of GPUs to use (default: 1)

NVIDIA L40s compatibility: This script loads models sequentially, deletes them from VRAM after use, and calls torch.cuda.empty_cache() to manage memory.

HuggingFace Hub: If a model is not present in model_dir, it will be downloaded from HuggingFace using the repo_id in agent_map. If authentication is required, set the HUGGINGFACE_TOKEN environment variable or login with `huggingface-cli login`.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

NEWSROOM_PATH = '../JUDGE-BENCH/data/newsroom/newsroom.json'
METRICS = ["Informativeness", "Relevance", "Fluency", "Coherence"]

# Hardcoded schema and prompt templates for each metric
eval_schema = {
    "Informativeness": {
        "prompt": "On a scale of 1 (low) to 5 (high), how well does the summary capture the key points of the article?\n\n{instance}\nRespond with a single integer.",
        "worst": 1,
        "best": 5
    },
    "Relevance": {
        "prompt": "On a scale of 1 (low) to 5 (high), are the details provided by the summary consistent with details in the article?\n\n{instance}\nRespond with a single integer.",
        "worst": 1,
        "best": 5
    },
    "Fluency": {
        "prompt": "On a scale of 1 (low) to 5 (high), are the individual sentences of the summary well-written and grammatical?\n\n{instance}\nRespond with a single integer.",
        "worst": 1,
        "best": 5
    },
    "Coherence": {
        "prompt": "On a scale of 1 (low) to 5 (high), do phrases and sentences of the summary fit together and make sense collectively?\n\n{instance}\nRespond with a single integer.",
        "worst": 1,
        "best": 5
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Profile LLMs for multi-dimensional evaluation using newsroom.json.")
    parser.add_argument('--model_dir', type=str, default='./saved_models', help='Directory containing model weights (default: ./saved_models)')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of agent names to profile (default: all)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for LLM inference (default: 8)')
    parser.add_argument('--output', type=str, default='profiling_results.json', help='Output file for profiling results (default: profiling_results.json)')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    return parser.parse_args()

def load_newsroom(path):
    data = read_json(path)
    return data['instances']

def get_metric_prompt(metric, instance_prompt):
    prompt_template = eval_schema[metric]["prompt"]
    return prompt_template.format(instance=instance_prompt)

def parse_llm_score(output):
    import re
    match = re.search(r"([1-5])", output)
    if match:
        return int(match.group(1))
    return None

def ensure_model_downloaded(model_id, model_dir):
    """
    Ensure the model is downloaded to model_dir/model_id. If not, download from HuggingFace.
    """
    from huggingface_hub import snapshot_download, HfApi, HfFolder
    local_dir = os.path.join(model_dir, model_id.replace('/', os.sep))
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        logging.info(f"Model {model_id} not found locally. Downloading from HuggingFace Hub to {local_dir}...")
        try:
            # Use token from env if available
            token = "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"
            snapshot_download(repo_id=model_id, local_dir=local_dir, token=token, resume_download=True)
        except Exception as e:
            logging.error(f"Failed to download model {model_id} from HuggingFace: {e}")
            raise
    else:
        logging.info(f"Model {model_id} found locally at {local_dir}.")
    return local_dir

def profile_llms(instances, agent_map, model_dir, models_to_run, batch_size, gpus):
    profiling_results = defaultdict(lambda: defaultdict(int))  # LLM -> metric -> score
    profiling_counts = defaultdict(lambda: defaultdict(int))   # LLM -> metric -> count

    for agent_name, model_id in agent_map.items():
        if models_to_run and agent_name not in models_to_run:
            continue
        logging.info(f"Profiling {agent_name} ({model_id})...")
        try:
            local_model_dir = ensure_model_downloaded(model_id, model_dir)
            llm = LLM(model=model_id, download_dir=model_dir, tensor_parallel_size=gpus, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_model_dir, trust_remote_code=True, use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load model/tokenizer for {agent_name}: {e}")
            continue
        prompts = []
        meta = []  # (instance_idx, metric)
        for idx, inst in enumerate(instances):
            instance_prompt = inst['instance']
            for metric in METRICS:
                prompt = get_metric_prompt(metric, instance_prompt)
                prompts.append(prompt)
                meta.append((idx, metric))
        all_outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"{agent_name} batches"):
            batch_prompts = prompts[i:i+batch_size]
            try:
                messages = [[{"role": "user", "content": p}] for p in batch_prompts]
                chat_prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
                sampling_params = SamplingParams(temperature=0.7, max_tokens=16)
                outputs = llm.generate(chat_prompts, sampling_params)
                batch_outputs = [o.outputs[0].text for o in outputs]
                all_outputs.extend(batch_outputs)
            except Exception as e:
                logging.error(f"LLM inference failed for batch {i//batch_size} of {agent_name}: {e}")
                all_outputs.extend([None]*len(batch_prompts))
        skipped = 0
        for (idx, metric), output in zip(meta, all_outputs):
            inst = instances[idx]
            human = inst['annotations'][metric]
            mean_human = human.get('mean_human')
            majority_human = human.get('majority_human', mean_human)  # fallback if missing
            llm_pred = parse_llm_score(output) if output is not None else None
            if llm_pred is None:
                logging.warning(f"Skipped: {agent_name} instance {idx} metric {metric} (output: {output})")
                skipped += 1
                continue
            if (abs(llm_pred - mean_human) <= 1) and (abs(llm_pred - majority_human) <= 1):
                profiling_results[agent_name][metric] += 1
            else:
                profiling_results[agent_name][metric] -= 1
            profiling_counts[agent_name][metric] += 1
        logging.info(f"{agent_name}: Skipped {skipped} / {len(meta)} prompts due to parse/inference errors.")
        # Free up GPU memory
        del llm
        del tokenizer
        torch.cuda.empty_cache()
    return profiling_results, profiling_counts

def save_profiling_results(results, counts, out_path='profiling_results.json'):
    summary = {}
    for agent, metrics in results.items():
        summary[agent] = {m: {"score": metrics[m], "count": counts[agent][m]} for m in METRICS}
    write_json(summary, out_path)
    logging.info(f"Profiling results saved to {out_path}")

    # Print table
    print("\nProfiling Summary Table:")
    header = ["LLM"] + METRICS
    print("\t".join(header))
    for agent in summary:
        row = [agent] + [str(summary[agent][m]["score"]) for m in METRICS]
        print("\t".join(row))

def main():
    args = parse_args()
    instances = load_newsroom(NEWSROOM_PATH)
    if args.models:
        models_to_run = set([m.strip() for m in args.models.split(",") if m.strip() in agent_map])
        if not models_to_run:
            logging.error(f"No valid models specified in --models. Available: {list(agent_map.keys())}")
            return
    else:
        models_to_run = None
    results, counts = profile_llms(instances, agent_map, args.model_dir, models_to_run, args.batch_size, args.gpus)
    save_profiling_results(results, counts, args.output)

if __name__ == "__main__":
    main() 