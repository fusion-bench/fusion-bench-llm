script_dir=$(dirname "$(realpath "$0")")
project_root=$(realpath "${script_dir}/../..")

rich-run fusion_bench_llm \
    method=simple_average \
    modelpool=Qwen2.5-1.5B_math_and_coder \
    merged_model_save_path=${project_root}/outputs/qwen2.5-1.5B_math_and_coder/simple_average \
    +merged_model_save_kwargs.save_tokenizer=true

rich-run fusion_bench_llm \
    method=simple_average \
    modelpool=Qwen2.5-7B_math_and_coder \
    merged_model_save_path=${project_root}/outputs/qwen2.5-7B_math_and_coder/simple_average \
    +merged_model_save_kwargs.save_tokenizer=true
