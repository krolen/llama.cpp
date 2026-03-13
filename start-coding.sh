export LLAMA_CACHE="unsloth/Qwen3.5-27B-GGUF"
./build/bin/llama-server \
    -hf unsloth/Qwen3.5-27B-GGUF:UD-Q4_K_XL \
    -ngl 99 \
    --flash-attn auto \
    --no-mmap \
    --port 8888 \
    --host 0.0.0.0 \
    -ctk q8_0 \
    -ctv q8_0 \
    --ctx-size 262144 \
    --reasoning off \
    --alias "mymodel" \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00


