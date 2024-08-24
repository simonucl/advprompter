MODELS=(
    "CohereForAI/aya-23-8B"
    "CohereForAI/aya-23-35B"
    "google/gemma-2-9b-it"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for MODEL in "${MODELS[@]}"
do
    DATA_DIR=output/prompter/llama3_opt/suffix_dataset
    MODEL_BASE_NAME=$(basename $MODEL)
    OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
    mkdir -p $OUTPUT_DIR

    python3 eval_jailbreak.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR
done