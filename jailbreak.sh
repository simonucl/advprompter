MODEL=CohereForAI/aya-23-8B
DATA_DIR=output/prompter/aya_opt/suffix_dataset
MODEL_BASE_NAME=$(basename $MODEL)
OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
mkdir -p $OUTPUT_DIR

python3 eval_jailbreak.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \