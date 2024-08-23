import torch
from utils import (
    Metrics,
    # check_jailbroken,
    column_names,
    dotdict,
    get_dataloader,
    log_data,
    read_csv_file,
    hit_rate_at_n,
    list_avg
)
import os
from collections import defaultdict
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
from agent import VllmAgent, GptAgent, CohereAgent
import argparse
from concurrent.futures import ThreadPoolExecutor

def check_jailbroken(seq, test_prefixes):
    jailbroken_list = [
        all([prefix not in text for prefix in test_prefixes]) for text in seq
    ]
    jailbroken_avg = list_avg(jailbroken_list)
    return jailbroken_avg, jailbroken_list

def _openai_completion_helper(api_key,
                              model_name,
                              instruct
                              ):
    agent = GptAgent(api_key=api_key, model_name=model_name)
    completion = agent.generate(instruct)
    return completion

@torch.no_grad()
def eval_suffix_datasets(agent, suffix_dataset_pth_dct, test_prefixes, output_dir, parallel=1):
    for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
        eval_suffix_dataset(
            agent=agent,
            suffix_dataset_key=suffix_dataset_key,
            suffix_dataset_pth=suffix_dataset_pth,
            test_prefixes=test_prefixes,
            output_dir=output_dir,
            parallel=parallel,
        )

@torch.no_grad()
def eval_suffix_dataset(agent, suffix_dataset_key, suffix_dataset_pth, test_prefixes, output_dir, parallel=1):
    # split = suffix_dataset_key
    logs = {}
    split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)

    df = pd.read_csv(suffix_dataset_pth)

    instructs = df['instruct'].tolist()
    full_instructs = df['full_instruct'].tolist()
    full_instructs = [[{'role': 'user', 'content': instruct}] for instruct in full_instructs]
    if isinstance(agent, VllmAgent):
        tokenizer = agent.tokenizer
        full_instructs = [tokenizer.apply_chat_template(instruct, add_generation_prompt=True, tokenize=False) for instruct in full_instructs]
    # eval_loader = get_dataloader(
    #     suffix_dataset_pth,
    #     shuffle=False,
    #     augment_target=False,
    #     batch_size=self.cfg.eval.batch_size,
    # )
    # eval_metrics = Metrics(prefix=split + "_eval/")

    instruct_jb_dict = defaultdict(list)
    if isinstance(agent, VllmAgent):
        completions = agent.generate(full_instructs)
    else:
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                completions = list(tqdm(executor.map(lambda x: _openai_completion_helper(agent.api_key, agent.model_name, x), full_instructs), total=len(full_instructs)))
        # completions = [agent.generate(instruct) for instruct in tqdm(full_instructs, total=len(full_instructs))]
        
        # --------- check jb for each trial
    jailbroken_avg, jailbroken_list = check_jailbroken(
        seq=completions, test_prefixes=test_prefixes
    )
    for instruct, jb in zip(instructs, jailbroken_list):
        instruct_jb_dict[instruct].append(jb)

    df['completion'] = completions
    df['jailbroken'] = jailbroken_list
    df.to_csv(f"{output_dir}/{suffix_dataset_key}_jailbreak.csv", index=False)
    # if basemodel_tf is not None:
    #     ppl_sum += basemodel_tf.perplexity.sum().item()

    # total_jailbroken = sum(
    #     eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"]
    # )
    # pbar.set_description(
    #     f"Evaluating {suffix_dataset_key} | Jailbroken {total_jailbroken}/{processed_samples}"
    # )

    # avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
    # avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = (
    #     float(
    #         sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
    #     )
    #     / processed_samples
    # )

    # tqdm.write(
    #     f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}"
    # )
    # tqdm.write(
    #     f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}"
    # )
    # tqdm.write(f" PPL: {float(ppl_sum) / processed_samples:.2f}")
    jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
    max_length = max(len(sublist) for sublist in jb_all)
    padded_list = [
        np.pad(sublist, (0, max_length - len(sublist)), "constant")
        for sublist in jb_all
    ]
    jb_stat_np = np.array(padded_list)
    for ti in range(1, jb_stat_np.shape[1] + 1):
        tqdm.write(
            f"{suffix_dataset_key} | hit rate @ {ti}: {hit_rate_at_n(jb_stat_np, ti)}"
        )
        logs[f"hit_rate@{ti}"] = hit_rate_at_n(jb_stat_np, ti)

    with open(f"{output_dir}/{suffix_dataset_key}_logs.txt", "w") as f:
        for k, v in logs.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--test_prefixes", type=str, default="data/test_prefixes.csv")
    argparser.add_argument("--output_dir", type=str, default="output")
    argparser.add_argument('--temperature', type=float, default=0.0)
    argparser.add_argument('--top_p', type=float, default=0.9)
    argparser.add_argument('--top_k', type=int, default=50)
    argparser.add_argument('--max_new_tokens', type=int, default=1024)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--parallel', type=int, default=1)

    args = argparser.parse_args()

    suffix_dataset_pth_dct = {
        # 'train': os.path.join(args.data_dir, 'train.csv'),
        'validation': os.path.join(args.data_dir, 'validation.csv'),
        'test': os.path.join(args.data_dir, 'test.csv'),
    }

    if "gpt" in args.model.lower():
        agent = GptAgent(api_key=os.environ["OPENAI_API_KEY"], model_name=args.model)
    elif "command" in args.model.lower():
        agent = CohereAgent(api_key=os.environ["COHERE_API_KEY"])
    else:
        model_kwargs = dict(data_parallel_size=1)
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_new_tokens,
            "seed": args.seed,
    }
        agent = VllmAgent(model_name=args.model, model_kwargs=model_kwargs, generation_kwargs=generation_kwargs)

    # Load the test prefixes
    # test_prefixes = read_csv_file(args.test_prefixes, column_names=["prefix"])["prefix"].tolist()
    test_prefixes = read_csv_file(args.test_prefixes)

    # Evaluate the test prefixes
    eval_suffix_datasets(agent, suffix_dataset_pth_dct, test_prefixes, args.output_dir, args.parallel)