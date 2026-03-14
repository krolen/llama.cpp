export LLAMA_CACHE="unsloth/Qwen3.5-27B-GGUF"
./build/bin/llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    -ngl 99 \
    --no-mmap \
    --port 8888 \
    --host 0.0.0.0 \
    --ctx-size 262144 \
    --reasoning off \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --fit on \
    --alias "mymodel" \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00

