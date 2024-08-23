
import torch
import os
from utils import (
    Metrics,
    check_jailbroken,
    column_names,
    dotdict,
    get_dataloader,
    log_data,
    read_csv_file,
    hit_rate_at_n,
)
from llm import LLM

from tqdm import tqdm
from collections import defaultdict
import csv
import warnings
import re
from omegaconf import DictConfig, OmegaConf

from sequence import MergedSeq, Seq, collate_fn
import numpy as np
import argparse

def batch_to_context(prompter, batch):
    model_map = dict(
        instruct=prompter,
        suffix=prompter,
        target=prompter,
        full_instruct=prompter,
    )
    context = dotdict()
    for key, model in model_map.items():
        if key in batch.keys():
            seq = Seq(
                text=batch[key],
                tokenizer=model.tokenizer,
                device=model.device,
            )
        else:
            seq = None
        context[key] = seq
    return context

@torch.no_grad()
def generate_suffix_datasets(prompter: LLM,
                            num_trials, 
                            batch_size,
                            max_new_tokens_list,
                            dataset_pth_dct,
                            suffix_dataset_dir):
    suffix_dataset_pth_dct = {}
    for dataset_key, dataset_pth in dataset_pth_dct.items():
        suffix_dataset = generate_suffix_dataset(
            dataset_key=dataset_key, dataset_pth=dataset_pth,
            prompter=prompter, num_trials=num_trials,
            batch_size=batch_size, max_new_tokens_list=max_new_tokens_list
        )
        suffix_dataset_pth = save_suffix_dataset(
            suffix_dataset, dir=suffix_dataset_dir
        )
        suffix_dataset_pth_dct[suffix_dataset.suffix_dataset_key] = (
            suffix_dataset_pth
        )
    return suffix_dataset_pth_dct

@torch.no_grad()
def generate_suffix_dataset(prompter: LLM,
                            num_trials, 
                            dataset_key, 
                            dataset_pth,
                            batch_size,
                            max_new_tokens_list):
    prompter.eval()

    data = []

    suffix_dataset_key = f"{dataset_key}"
    eval_loader = get_dataloader(
        data_pth=dataset_pth,
        shuffle=False,
        augment_target=False,
        batch_size=batch_size,
    )
    pbar_batches = tqdm(eval_loader)
    pbar_batches.set_description(f"Generating suffix dataset {suffix_dataset_key}")
    for batch in pbar_batches:
        context = batch_to_context(prompter, batch)
        instruct = context.instruct
        target = context.target
        batch_data = []
        for max_new_tokens in max_new_tokens_list:
            trial_data = []
            for trial in range(num_trials):
                prompter_ar = prompter.generate_autoregressive(
                    key="suffix",
                    max_new_tokens=max_new_tokens,
                    instruct=instruct,
                )
                suffix = prompter_ar.response_sample

                full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(
                    merge_dtype="ids"
                )

                assert instruct.bs == target.bs == suffix.bs
                datapoint = []
                for i in range(instruct.bs):
                    datapoint.append(
                        (
                            instruct.text[i],
                            target.text[i],
                            suffix.text[i],
                            full_instruct.text[i],
                        )
                    )
                trial_data.append(datapoint)
            batch_data.append(trial_data)

        for i in range(instruct.bs):
            for j in range(len(max_new_tokens_list)):
                for k in range(num_trials):
                    data.append(batch_data[j][k][i])

    suffix_dataset = dotdict(
        data=data,
        fields=["instruct", "target", "suffix", "full_instruct"],
        suffix_dataset_key=suffix_dataset_key,
    )

    return suffix_dataset

@torch.no_grad()
def save_suffix_dataset(suffix_dataset, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    suffix_dataset_pth = os.path.join(
        dir,
        suffix_dataset.suffix_dataset_key + ".csv",
    )
    tqdm.write(
        f" Saving {suffix_dataset.suffix_dataset_key} to {suffix_dataset_pth}"
    )
    with open(suffix_dataset_pth, "w") as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(suffix_dataset.fields)
        csvwriter.writerows(suffix_dataset.data)
    return suffix_dataset_pth

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_param_path", type=str, default="gpt2")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    max_new_tokens_list = [30, 50]
    dataset_pth_dct = {
        'train': os.path.join(args.data_dir, 'harmful_behaviors', 'dataset', 'train.csv'),
        'validation': os.path.join(args.data_dir, 'harmful_behaviors', 'dataset', 'validation.csv'),
        'test': os.path.join(args.data_dir, 'harmful_behaviors', 'dataset', 'test.csv'),
    }
    suffix_dataset_dir = os.path.join(args.output_dir, 'suffix_dataset')

    prompter_config = OmegaConf.load(args.model_param_path)
    OmegaConf.set_struct(prompter_config, True)

    prompter = LLM(prompter_config, verbose=True)
    suffix_dataset_pth_dct = generate_suffix_datasets(prompter, args.num_trials, args.batch_size, max_new_tokens_list, dataset_pth_dct, suffix_dataset_dir)
