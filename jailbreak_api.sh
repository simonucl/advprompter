MODELS=(
    "gpt-4o-mini"
    "gpt-4o"
    "command-r"
    "command-r-plus"
)
BASE_DIR=output/prompter/llama3_opt
for MODEL in "${MODELS[@]}"
do
    DATA_DIR=$BASE_DIR/suffix_dataset
    MODEL_BASE_NAME=$(basename $MODEL)
    OUTPUT_DIR=$DATA_DIR/$MODEL_BASE_NAME
    mkdir -p $OUTPUT_DIR

    python3 eval_jailbreak.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --parallel 16
done