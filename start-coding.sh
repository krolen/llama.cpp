export LLAMA_CACHE="/home/kot/unsloth"

# stable qwen 27b dense
#./build/bin/llama-server \
#    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
#    -ngl 99 \
#    --no-mmap \
#    --port 8888 \
#    --host 0.0.0.0 \
#    --ctx-size 262144 \
#    --reasoning off \
#    --kv-unified \
#    --cache-type-k q8_0 --cache-type-v q8_0 \
#    --flash-attn on --fit on \
#    --alias "mymodel" \
#    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
#    --jinja \
#    --ubatch-size 1024 --batch-size 4096 \
#    --ctx-checkpoints 0 \
#    --swa-full \
#    --cont-batching \

#./build/bin/llama-server \
#    -m $LLAMA_CACHE/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
#    --mmproj $LLAMA_CACHE/HauhauCS/mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf \
#    -ngl 99 \
#    --no-mmap \
#    --port 8888 \
#    --host 0.0.0.0 \
#    --ctx-size 262144 \
#    --reasoning off \
#    --kv-unified \
#    --cache-type-k q8_0 --cache-type-v q8_0 \
#    --flash-attn on --fit on \
#    --alias "mymodel" \
#    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00 \
#    --jinja \
#    --ubatch-size 1024 --batch-size 4096 \
#    --ctx-checkpoints 0 \
#    --swa-full \
#    --cont-batching \
#
#    --ctx-size 262144 \
#    --chat-template chatml \
#    --chat-template-file qwen3.5_chat_template.jinja


# gemma 4
#    --reasoning off \
#    -hf ggml-org/gemma-4-31B-it-GGUF:Q4_K_M \
#    --no-mmap \
#    --ctx-size 181072 \
#    --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.00 \

./build/bin/llama-server \
    -hf unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL \
    -ngl all --flash-attn on --fit on \
    --no-mmap \
    --port 8888 \
    --host 0.0.0.0 \
    --ctx-size 65536 \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --alias "mymodel" \
    --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.00 \
    --jinja \
    --ctx-checkpoints 0 \
    --cont-batching \



# if one need to download  models

#huggingface-cli download HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive \
#  --include "mmproj-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-f16.gguf" \
#  --local-dir ~/unsloth/HauhauCS \
#  --local-dir-use-symlinks False
#
#huggingface-cli download HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive \
#  --include "*Q4_K_M.gguf" \
#  --local-dir ~/unsloth/HauhauCS \
#  --local-dir-use-symlinks False
