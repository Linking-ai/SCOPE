export CUDA_VISIBLE_DEVICES=2

method="PyramidInfer" # Support PyramidKV, SnapKV, PyramidInfer, H2O, StreamingLLM, ALLKV
max_capacity_prompts=2048 # 2048 in paper
decoding_recent_size=256 # 256 in paper
decoding_window_size=512 # 512,1024 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
model_path=/data1/wangzhenglin/speculative_decoding/LLM/Meta-Llama-3.1-8B-Instruct
K=30

for decoding_metric in "pyramidinfer"
do
    echo "Using method: ${method}, decoding_metric: ${decoding_metric}"
    source_path=results_longgenbench_4K/
    save_dir="${source_path}${method}+${decoding_metric}_decoding_window_${decoding_window_size}_recent_window_${decoding_recent_size}" # path to result save_dir
    python3 run_longgenbench.py \
        --method ${method} \
        --model_path ${model_path} \
        --max_capacity_prompts ${max_capacity_prompts} \
        --attn_implementation ${attn_implementation} \
        --save_dir ${save_dir} \
        --use_cache True \
        --K ${K}\
        --decoding_window_size ${decoding_window_size} \
        --decoding_recent_size ${decoding_recent_size} \
        --decoding_metric ${decoding_metric} \

done
