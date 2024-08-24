OUTPUT_DIR=output/prompter/llama3_opt
mkdir -p $OUTPUT_DIR
python3 gen_advprompt.py \
    --model_param_path conf/prompter/llama3_opt.yaml --output_dir $OUTPUT_DIR