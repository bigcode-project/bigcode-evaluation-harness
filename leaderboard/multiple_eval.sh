cd /fsx/loubna/code/dev/leader/bigcode-evaluation-harness/leaderboard
langs=(js java cpp swift php d jl lua r rkt rb rs)

model=NewHope
org=/fsx/loubna/models/
out_path=/fsx/loubna/code/dev/bigcode-evaluation-harness/generations_newhope

for lang in "${langs[@]}"; do
    if [ "$lang" == "py" ]; then
        task=humaneval
    else
        task=multiple-$lang
    fi
    echo "Submitting task $task"
    sbatch -J "eval-$model-$task" /fsx/loubna/code/dev/leader/bigcode-evaluation-harness/leaderboard/multiple_eval.slurm "$model" "$task" "$org" "$out_path"
done