#!/bin/bash

# Variables
model_name="Salesforce/codegen-350M-mono"

# Function to get the current timestamp
timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}
curr_ts=$(timestamp)

# Create the output directory
mkdir -p outputs/$curr_ts-lsf
output_dir=outputs/$curr_ts-lsf

# Create the job script
cat << EOF > $output_dir/job_script.sh
#!/bin/bash
#BSUB -J $curr_ts-$model_name
#BSUB -q waic-short 
#BSUB -n 1
#BSUB -gpu "num=1:j_exclusive=yes:gmem=40G"
#BSUB -R rusage[mem=64GB]
#BSUB -oo $output_dir/stdout.txt
#BSUB -eo $output_dir/stderr.txt

echo "Timestamp: $curr_ts"
cd ~/repos/bigcode-evaluation-harness
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "pwd=$PWD"

knockknock slack --webhook-url https://hooks.slack.com/services/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 --channel Slackbot \
accelerate launch --num_machines=1 --multi_gpu main.py \
--model=$model_name --tasks=program_repair,humaneval \
--limit=100 --max_length_generation=512 --temperature=0.2 --do_sample=True --n_samples=200 --batch_size=200 \
--allow_code_execution --trust_remote_code --save_generations --save_references --use_auth_token
EOF

# Set execute permissions on the job script
chmod +x $output_dir/job_script.sh

# Submit the job
bsub < $output_dir/job_script.sh