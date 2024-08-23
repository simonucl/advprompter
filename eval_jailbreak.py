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

def check_jailbroken(seq, test_prefixes):
    jailbroken_list = [
        all([prefix not in text for prefix in test_prefixes]) for text in seq.text
    ]
    jailbroken_avg = list_avg(jailbroken_list)
    return jailbroken_avg, jailbroken_list


@torch.no_grad()
def eval_suffix_datasets(agent, suffix_dataset_pth_dct, test_prefixes, output_dir):
    for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
        eval_suffix_dataset(
            model=agent,
            suffix_dataset_key=suffix_dataset_key,
            suffix_dataset_pth=suffix_dataset_pth,
            test_prefixes=test_prefixes,
            output_dir=output_dir,
        )

@torch.no_grad()
def eval_suffix_dataset(agent, suffix_dataset_key, suffix_dataset_pth, test_prefixes, output_dir):
    # split = suffix_dataset_key
    logs = {}
    split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)

    df = pd.read_csv(suffix_dataset_pth)

    instructs = df['instruct'].tolist()
    full_instructs = df['full_instruct'].tolist()

    # eval_loader = get_dataloader(
    #     suffix_dataset_pth,
    #     shuffle=False,
    #     augment_target=False,
    #     batch_size=self.cfg.eval.batch_size,
    # )
    # eval_metrics = Metrics(prefix=split + "_eval/")

    instruct_jb_dict = defaultdict(list)
    completions = agent.generate(full_instructs)
        
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
    args = argparser.parse_args()

    suffix_dataset_pth_dct = {
        'train': os.path.join(args.data_dir, 'train.csv'),
        'validation': os.path.join(args.data_dir, 'validation.csv'),
        'test': os.path.join(args.data_dir, 'test.csv'),
    }

    if "gpt" in args.model.lower():
        agent = GptAgent(api_key=os.environ["OPENAI_API_KEY"], model_name=args.model)
    elif "command" in args.model.lower():
        agent = CohereAgent(api_key=os.environ["COHERE_API_KEY"], model_name=args.model)
    else:
        model_kwargs = dict()
        generation_kwargs = dict()
        agent = VllmAgent(model_name=args.model, model_kwargs=model_kwargs, generation_kwargs=generation_kwargs)

    # Load the test prefixes
    test_prefixes = read_csv_file(args.test_prefixes, column_names=["prefix"])["prefix"].tolist()

    # Evaluate the test prefixes
    eval_suffix_datasets(agent, suffix_dataset_pth_dct, test_prefixes, args.output_dir)