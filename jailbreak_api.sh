MODEL=gpt-4o-mini
DATA_DIR=output/prompter/aya_opt/suffix_dataset
MODEL_BASE_NAME=$(basename $MODEL)
OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
mkdir -p $OUTPUT_DIR

python3 eval_jailbreak.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --parallel 16

MODEL=command-r
DATA_DIR=output/prompter/aya_opt/suffix_dataset
MODEL_BASE_NAME=$(basename $MODEL)
OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
mkdir -p $OUTPUT_DIR

python3 eval_jailbreak.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --parallel 16

MODEL=command-r-plus
DATA_DIR=output/prompter/aya_opt/suffix_dataset
MODEL_BASE_NAME=$(basename $MODEL)
OUTPUT_DIR=output/prompter/aya_opt/$MODEL_BASE_NAME
mkdir -p $OUTPUT_DIR

python3 eval_jailbreak.py \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --parallel 16