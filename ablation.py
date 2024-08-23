
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
# from omegaconf import DictConfig, OmegaConf

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

# @torch.no_grad()
# def eval_suffix_datasets(suffix_dataset_pth_dct):
#     for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
#         eval_suffix_dataset(
#             suffix_dataset_key=suffix_dataset_key,
#             suffix_dataset_pth=suffix_dataset_pth,
#         )

# @torch.no_grad()
# def eval_suffix_dataset(suffix_dataset_key, suffix_dataset_pth):
#     self.prompter.eval()
#     self.target_llm.eval()

#     # split = suffix_dataset_key
#     split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)

#     eval_loader = get_dataloader(
#         suffix_dataset_pth,
#         shuffle=False,
#         augment_target=False,
#         batch_size=self.cfg.eval.batch_size,
#     )
#     eval_metrics = Metrics(prefix=split + "_eval/")

#     instruct_jb_dict = defaultdict(list)
#     processed_samples, ppl_sum = 0, 0
#     pbar = tqdm(eval_loader)
#     pbar.set_description(
#         f"Evaluating suffix dataset {suffix_dataset_key} | Jailbroken 0/0 | Success 0/0"
#     )
#     for batch_idx, batch in enumerate(pbar):
#         context = self.batch_to_context(batch)
#         instruct = context.instruct
#         suffix = context.suffix
#         full_instruct = context.full_instruct
#         target = context.target
#         target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
#             cfg=self.cfg,
#             instruct=instruct,
#             suffix=suffix,
#             full_instruct=full_instruct,
#             target=target,
#             prompter=self.prompter,
#             target_llm=self.target_llm,
#             generate_target_llm_response=True,
#         )

#         # --------- check jb for each trial
#         _, jailbroken_list = check_jailbroken(
#             seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes
#         )
#         instruct = instruct
#         assert instruct.bs == len(jailbroken_list)
#         instruct_text = instruct.text
#         for i in range(instruct.bs):
#             instruct_jb_dict[instruct_text[i]].append(jailbroken_list[i])
#         # -----------

#         log_data(
#             log_table=None,
#             metrics=eval_metrics,
#             step=self.step,
#             split=split,
#             batch_idx=batch_idx,
#             test_prefixes=self.test_prefixes,
#             affirmative_prefixes=self.affirmative_prefixes,
#             batch_size=self.cfg.eval.batch_size,
#             log_sequences_to_wandb=False,
#             log_metrics_to_wandb=False,
#             target_llm_tf=target_llm_tf,
#             target_llm_ar=target_llm_ar,
#             basemodel_tf=basemodel_tf,
#         )
#         processed_samples += instruct.bs
#         if basemodel_tf is not None:
#             ppl_sum += basemodel_tf.perplexity.sum().item()

#         total_jailbroken = sum(
#             eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"]
#         )
#         pbar.set_description(
#             f"Evaluating {suffix_dataset_key} | Jailbroken {total_jailbroken}/{processed_samples}"
#         )

#     avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
#     avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = (
#         float(
#             sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
#         )
#         / processed_samples
#     )

#     tqdm.write(
#         f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}"
#     )
#     tqdm.write(
#         f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}"
#     )
#     tqdm.write(f" PPL: {float(ppl_sum) / processed_samples:.2f}")
#     jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
#     max_length = max(len(sublist) for sublist in jb_all)
#     padded_list = [
#         np.pad(sublist, (0, max_length - len(sublist)), "constant")
#         for sublist in jb_all
#     ]
#     jb_stat_np = np.array(padded_list)
#     for ti in range(1, jb_stat_np.shape[1] + 1):
#         tqdm.write(
#             f"{suffix_dataset_key} | hit rate @ {ti}: {hit_rate_at_n(jb_stat_np, ti)}"
#         )

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_param_path", type=str, default="gpt2")
    parser.add_argument("--data_idr", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    max_new_tokens_list = [30, 50]
    dataset_pth_dct = {
        'train': os.path.join(args.data_idr, 'harmful_behaviors', 'dataset', 'train.csv'),
        'validation': os.path.join(args.data_idr, 'harmful_behaviors', 'dataset', 'validation.csv'),
        'test': os.path.join(args.data_idr, 'harmful_behaviors', 'dataset', 'test.csv'),
    }
    suffix_dataset_dir = os.path.join(args.output_dir, 'suffix_dataset')

    prompter_config = OmegaConf.load(args.model_param_path)
    OmegaConf.set_struct(prompter_config, True)

    prompter = LLM(prompter_config, verbose=True)
    suffix_dataset_pth_dct = generate_suffix_datasets(prompter, args.num_trials, args.batch_size, max_new_tokens_list, dataset_pth_dct, suffix_dataset_dir)
