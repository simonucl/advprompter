MODELS=(
    "CohereForAI/aya-23-8B"
    "CohereForAI/aya-23-35B"
    "command-r-08-2024"
    "command-r"
    "command-r-plus"
    "command-r-plus-08-2024"
    "google/gemma-2-9b-it"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    # "simonycl/aya-advprompter"
    # "simonycl/aya-advprompter-checkpoint-50"
    # "simonycl/aya-advprompter-checkpoint-100"
    # "simonycl/aya-advprompter-checkpoint-150"
)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
BASE_DIR=output/prompter/aya_opt

for MODEL in "${MODELS[@]}"
do
    DATA_DIR=$BASE_DIR/suffix_dataset
    MODEL_BASE_NAME=$(basename $MODEL)
    OUTPUT_DIR=$DATA_DIR/$MODEL_BASE_NAME
    mkdir -p $OUTPUT_DIR

    python3 eval_jailbreak.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR
done