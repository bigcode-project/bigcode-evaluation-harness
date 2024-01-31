
#accelerate launch  main.py \
python  ~/src/bigcode-evaluation-harness/main.py \
  --model ${1} \
  --tasks humaneval \
  --max_length_generation 512 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 50 \
  --max_memory_per_gpu auto \
  --batch_size 32 \
  --allow_code_execution \
  --save_generations \
  --precision fp16 \
  --modeltype='nm' \
  --metric_output_path ${1}/evaluation_results.json
  #--limit 50 \
