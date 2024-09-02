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
from strong_reject.evaluate import evaluate

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

def _cohere_completion_helper(api_key,
                              model_name,
                              instruct
                              ):
    agent = CohereAgent(api_key=api_key, model_name=model_name)
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
    if 'full_instruct' not in df.columns:
        df['full_instruct'] = df['instruct']
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
    elif isinstance(agent, GptAgent):
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                completions = list(tqdm(executor.map(lambda x: _openai_completion_helper(os.environ["OPENAI_API_KEY"], agent.model_name, x), full_instructs), total=len(full_instructs)))
        # completions = [agent.generate(instruct) for instruct in tqdm(full_instructs, total=len(full_instructs))]
    elif isinstance(agent, CohereAgent):
        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                completions = list(tqdm(executor.map(lambda x: _cohere_completion_helper(os.environ["COHERE_API_KEY"], agent.model_name, x), full_instructs), total=len(full_instructs)))
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")
    
        # completions = [agent.generate(instruct) for instruct in tqdm(full_instructs, total=len(full_instructs))]
        # --------- check jb for each trial
    df['completion'] = completions

    evaluate_methods = ['string_matching', 'strongreject_rubric']
    jailbroken_dict = defaultdict(list)
    def process_instruct_completion(args):
        full_instruct, completion = args
        scores = evaluate(full_instruct, completion, evaluate_methods)
        return {method: {'jailbroken': bool(score['score'] == 1.0), 'score': score['score']} for method, score in zip(evaluate_methods, scores)}

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        results = list(tqdm(
            executor.map(process_instruct_completion, zip(full_instructs, completions)),
            total=len(full_instructs),
            desc="Evaluating Jailbreak"
        ))

    for method in evaluate_methods:
        jailbroken_dict[method] = [result[method] for result in results]

    for method in evaluate_methods:
        df[f'jailbroken_{method}'] = [result['jailbroken'] for result in jailbroken_dict[method]]
        df[f'jailbroken_score_{method}'] = [result['score'] for result in jailbroken_dict[method]]
        instruct_jb_dict = defaultdict(list)
        for instruct, jb in zip(instructs, jailbroken_dict[method]):
            instruct_jb_dict[instruct].append(jb['jailbroken'])

        jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
        max_length = max(len(sublist) for sublist in jb_all)
        padded_list = [
            np.pad(sublist, (0, max_length - len(sublist)), "constant")
            for sublist in jb_all
        ]
        jb_stat_np = np.array(padded_list)
        for ti in range(1, jb_stat_np.shape[1] + 1):
            hit_rate = hit_rate_at_n(jb_stat_np, ti)
            tqdm.write(
                f"{suffix_dataset_key} | {method} | hit rate @ {ti}: {hit_rate}"
            )
            logs[f"{method}_hit_rate@{ti}"] = hit_rate

    df.to_csv(f"{output_dir}/{suffix_dataset_key}_jailbreak.csv", index=False)

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
        # 'validation': os.path.join(args.data_dir, 'validation.csv'),
        'test': os.path.join(args.data_dir, 'test.csv'),
    }

    if "gpt" in args.model.lower():
        agent = GptAgent(api_key=os.environ["OPENAI_API_KEY"], model_name=args.model)
    elif "command" in args.model.lower():
        agent = CohereAgent(api_key=os.environ["COHERE_API_KEY"], model_name=args.model)
    else:
        model_kwargs = dict(tensor_parallel_size=2,max_model_len=2048)
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
