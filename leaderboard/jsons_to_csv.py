import argparse
import pandas as pd
import json
import os
import glob


parser = argparse.ArgumentParser(description='Process metric files')
parser.add_argument('--metrics_path', type=str, required=True, help='Path where metric files are stored')
parser.add_argument('--csv_name', type=str, required=True, help='Name of csv file to be created')
args = parser.parse_args()


# List of valid tasks
valid_tasks = ["humaneval"] + ["multiple-" + lang for lang in ["js", "java", "cpp", "swift", "php", "d", "jl", "lua", "r", "rkt", "rb", "rs"]]

df = pd.DataFrame(columns=["task", "pass@1"])

# Iterate over all .json files in the metrics_path
for json_file in glob.glob(os.path.join(args.metrics_path, '*.json')):

    # Extract task from file name
    task = os.path.splitext(os.path.basename(json_file))[0].split('_')[1]
    if task not in valid_tasks:
        print(f"Skipping invalid task: {task}")
        continue

    with open(json_file, 'r') as f:
        data = json.load(f)

    pass_at_1 = data.get(task, {}).get("pass@1", None)
    df = df.append({"task": task, "pass@1": pass_at_1}, ignore_index=True)

print(df)
df.to_csv(args.csv_name, index=False)
