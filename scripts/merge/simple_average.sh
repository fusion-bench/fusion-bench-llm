script_dir=$(dirname "$(realpath "$0")")
project_root=$(realpath "${script_dir}/../..")

fusion_bench --config-dir ${project_root}/config \
    method=linear/simple_average_for_llama \
    modelpool=Qwen2.5-1.5B_math_and_coder \
    merged_model_save_path=${project_root}/outputs/qwen2.5-1.5B_math_and_coder/simple_average \
    +merged_model_save_kwargs.save_tokenizer=true
