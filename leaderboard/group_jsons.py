import argparse
import pandas as pd
import json
import os
import glob


parser = argparse.ArgumentParser(description='Process metric files')
parser.add_argument('--metrics_path', type=str, required=True, help='Path where metric files are stored')
parser.add_argument('--model', type=str, required=True, help='Name of the model')
parser.add_argument('--org', type=str, required=True, help='Organization/user hosting the model')
parser.add_argument('--username', type=str, required=True, help='Your HF username')
args = parser.parse_args()


# List of valid tasks
valid_tasks = ["humaneval"] + ["multiple-" + lang for lang in ["js", "java", "cpp", "swift", "php", "d", "jl", "lua", "r", "rkt", "rb", "rs"]]

final_results = {"results": [], "meta": {"model": f"{args.org}/{args.model}"}}

# Iterate over all .json files in the metrics_path
for json_file in glob.glob(os.path.join(args.metrics_path, '*.json')):

    # Extract task from file name
    print(f"Processing {json_file}")
    task = os.path.splitext(os.path.basename(json_file))[0].split('_')[1]
    if task not in valid_tasks:
        print(f"Skipping invalid task: {task}")
        continue

    with open(json_file, 'r') as f:
        data = json.load(f)

    pass_at_1 = data.get(task, {}).get("pass@1", None)
    output = {"task": task, "pass@1": pass_at_1}
    final_results["results"].append(output)
    

with open(f"{args.org}_{args.model}_{args.username}.json", 'w') as f:
    json.dump(final_results, f)

print(f"Saved {args.org}_{args.model}_{args.username}.json")