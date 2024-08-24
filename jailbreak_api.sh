MODELS=(
    "gpt-4o-mini"
    "gpt-4o"
    "command-r"
    "command-r-plus"
)
for MODEL in "${MODELS[@]}"
do
    DATA_DIR=output/prompter/llama3_opt/suffix_dataset
    MODEL_BASE_NAME=$(basename $MODEL)
    OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
    mkdir -p $OUTPUT_DIR

    python3 eval_jailbreak.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --parallel 16
done