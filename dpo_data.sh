MODELS=(
    "gpt-4o"
    "CohereForAI/aya-23-8B"
)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
BASE_DIR=output/prompter/aya_opt

# for MODEL in "${MODELS[@]}"
# do
#     DATA_DIR=$BASE_DIR/suffix_dataset
#     MODEL_BASE_NAME=$(basename $MODEL)
#     OUTPUT_DIR=$DATA_DIR/${MODEL_BASE_NAME}_advbench
#     mkdir -p $OUTPUT_DIR
# 
#     python3 eval_jailbreak.py \
#         --model $MODEL \
#         --data_dir $DATA_DIR \
#         --output_dir $OUTPUT_DIR \
# 	--parallel 32
# done

for MODEL in "${MODELS[@]}"
do
    DATA_DIR=data/harmful_behaviors/dataset
    MODEL_BASE_NAME=$(basename $MODEL)
    OUTPUT_DIR=$DATA_DIR/${MODEL_BASE_NAME}_advbench
    mkdir -p $OUTPUT_DIR

    python3 eval_jailbreak.py \
        --model $MODEL \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
	--parallel 32
done
