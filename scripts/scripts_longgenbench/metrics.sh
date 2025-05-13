# results_dir="results_longgenbench_4K/ALLKV+None_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"

# results_dir="results_longgenbench_4K/H2O+h2o_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+slide_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+adaptive_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+discontinuous_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/PyramidInfer+pyramidinfer_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/StreamingLLM+slm_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"

# results_dir="results_longgenbench_4K/H2O+h2o_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+slide_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+adaptive_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/H2O+discontinuous_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/PyramidInfer+pyramidinfer_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_4K/StreamingLLM+slm_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"


# results_dir="results_longgenbench_8K/ALLKV+None_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"

# results_dir="results_longgenbench_8K/H2O+h2o_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+slide_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+adaptive_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+discontinuous_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/StreamingLLM+slm_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/PyramidInfer+pyramidinfer_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_4096"

# results_dir="results_longgenbench_8K/H2O+h2o_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+slide_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+adaptive_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/H2O+discontinuous_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/StreamingLLM+slm_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"
# results_dir="results_longgenbench_8K/PyramidInfer+pyramidinfer_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_4096"

# results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+slide_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+adaptive_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+discontinuous_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"

# results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+slide_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+adaptive_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"
results_dir="results_longgenbench_gsm8k_plug_in/ALLKV+discontinuous_decoding_window_1024_recent_window_256/meta-llama-3.1-8b-instruct_2048"


python3 eval_longgenbench.py \
    --results_dir ${results_dir}
