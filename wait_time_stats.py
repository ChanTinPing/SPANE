import os
import sys
import re
import json
import pandas as pd
from statistics import mean

# Define file paths
INPUT_PATH  = 'result/wait_time/result.txt'
OUTPUT_PATH = 'result/wait_time/result.csv'

# Check that input directory exists
input_dir = os.path.dirname(INPUT_PATH)
if not os.path.isdir(input_dir):
    sys.exit(f"Error: directory '{input_dir}' does not exist. No data to process.")

# Load and parse the text file
pattern = re.compile(r'(\w+)\s+Trial_id:\s+(\w+),\s+Result:\s+([\d.]+)')
records = []
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.match(line.strip())
        if m:
            method, trial_id, result = m.groups()
            records.append({
                'method': method,
                'trial_id': trial_id,
                'result': float(result)
            })

# Ensure we have data
if not records:
    sys.exit(f"Error: no valid records found in '{INPUT_PATH}'.")

# Create DataFrame
df = pd.DataFrame(records)

def compute_trimmed_mean(results):
    """Drop lowest 2 and highest 2, then compute mean (rounded to int)."""
    s = sorted(results)
    if len(s) > 4:
        return int(round(mean(s[2:-2])))
    return pd.NA

def compute_mean_of_top3(grp, method):
    """
    For each method, read models/{method}/results.json,
    take the first 3 trial_ids as top3, then average their results.
    """
    json_path = f"models/{method}/results.json"
    if not os.path.isfile(json_path):
        sys.exit(f"Error: top3 JSON file '{json_path}' not found for method '{method}'.")
    with open(json_path, 'r', encoding='utf-8') as jf:
        data = json.load(jf)
    top3_ids = [entry['trial_id'] for entry in data][:3]
    selected = grp[grp['trial_id'].isin(top3_ids)]['result'].tolist()
    if len(selected) == 3:
        return int(round(mean(selected)))
    return pd.NA

# Aggregate metrics by method
summary_rows = []
for method, group in df.groupby('method'):
    trimmed = compute_trimmed_mean(group['result'].tolist())
    top3    = compute_mean_of_top3(group, method)
    summary_rows.append({
        'method': method,
        'trimmed_mean': trimmed,
        'mean_of_top3': top3
    })

# Build summary DataFrame and save to CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f"Finished: results saved to '{OUTPUT_PATH}'.")
