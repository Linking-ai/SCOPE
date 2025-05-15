# results_dir="results_infinitebench/H2O+h2o_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_infinitebench/H2O+slide_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_infinitebench/H2O+adaptive_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_infinitebench/H2O+discontinuous_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_infinitebench/StreamingLLM+slm_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
# results_dir="results_infinitebench/PyramidInfer+pyramidinfer_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"
results_dir="results_infinitebench/ALLKV+None_decoding_window_512_recent_window_256/meta-llama-3.1-8b-instruct_2048"

python3 eval_infinitebench.py \
    --results_dir ${results_dir}
