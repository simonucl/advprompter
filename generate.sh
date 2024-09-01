OUTPUT_DIR=output/prompter/aya_opt
mkdir -p $OUTPUT_DIR
python3 gen_advprompt.py \
    --model_param_path conf/prompter/aya_opt.yaml --output_dir $OUTPUT_DIR