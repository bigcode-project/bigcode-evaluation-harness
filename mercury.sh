accelerate  launch --main_process_port 29501  main.py  \
    --model deepseek-ai/deepseek-coder-6.7b-base   \
    --load_in_4bit   \
    --limit 256 \
    --max_length_generation 1024   \
    --tasks mercury   \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 6   \
    --allow_code_execution

accelerate  launch --main_process_port 29502  main.py  \
    --model deepseek-ai/deepseek-coder-6.7b-instruct   \
    --load_in_4bit   \
    --limit 256 \
    --max_length_generation 1024   \
    --tasks mercury   \
    --n_samples 5  \
    --temperature 0.2  \
    --batch_size 6   \
    --allow_code_execution

