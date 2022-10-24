echo setup on Linux machine with torch preinstalled, execute inside bigcode-evaluation-harness
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
pip install -r requirements.txt

accelerate config
echo changing ulimit to 50000 avoid too many open files error, it happens when there are many cpu workers
ulimit -n 50000

echo downloading MBPP generations from the hub and copying them to current directory
cd ..
git clone https://huggingface.co/datasets/loubnabnl/code-generations-bigcode
cp code-generations-bigcode/codeparrotdedup-08/generations.json bigcode-evaluation-harness/

echo small test run on 4 tasks
cd bigcode-evaluation-harness
accelerate launch  main.py   --tasks mbpp   --prompt_type_mbpp "incoder" --allow_code_execution=True --evaluation_only True   --model codeparrotdedup-08  --num_tasks_mbpp 4

echo running evaluation on all 500 tasks
accelerate launch  main.py   --tasks mbpp   --prompt_type_mbpp "incoder" --allow_code_execution=True --evaluation_only True   --model codeparrotdedup-08
