echo setup file for evaluation only mode, cloning repo
git clone https://github.com/loubnabnl/bigcode-evaluation-harness.git
cd bigcode-evaluation-harness

echo setup on Linux machine with torch preinstalled
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
cd bigcode-evaluation-harness
pip install -r requirements.txt
cd ..

accelerate config
echo changing ulimit to 50000 avoid too many open files error
ulimit -n 50000

echo downloading MBPP generations from the hub and copying them to current directory
git clone https://huggingface.co/datasets/loubnabnl/code-generations-bigcode
cp code-generations-bigcode/anylicense-02/generations.json bigcode-evaluation-harness/

echo small test run on 4 tasks
cd bigcode-evaluation-harness
accelerate launch  main.py   --tasks mbpp   --prompt_type_mbpp "incoder" --allow_code_execution=True --evaluation_only True   --model gpt2-any-license-02  --num_tasks_mbpp 4

echo running evaluation on all 500 tasks
accelerate launch  main.py   --tasks mbpp   --prompt_type_mbpp "incoder" --allow_code_execution=True --evaluation_only True   --model gpt2-any-license-02
