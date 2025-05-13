import os
import sys
import re
import pandas as pd

# Define paths
RESULT_PATH       = 'result/flex/result.txt'
BASELINE_PATH     = 'result/flex/result_baseline.txt'
OUTPUT_CSV_PATH   = 'result/flex/result.csv'

# Check that input directory exists
input_dir = os.path.dirname(RESULT_PATH)
if not os.path.isdir(input_dir):
    sys.exit(f"Error: directory '{input_dir}' does not exist. No data to process.")

# --- 1. Read and parse SPANE results ---
spane_records = []
pattern_spane = re.compile(r'SPANE\s+PM_num:\s*(\d+),\s*Trial_id:\s*(\w+),\s*Result:\s*([\d.]+)')
with open(RESULT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern_spane.match(line.strip())
        if m:
            pm_num = int(m.group(1))
            result = float(m.group(3))
            spane_records.append({'PM_num': pm_num, 'result': result})

if not spane_records:
    sys.exit(f"Error: no valid SPANE records found in '{RESULT_PATH}'.")

# Compute average SPANE result per PM_num
df_spane = pd.DataFrame(spane_records)
df_spane_avg = df_spane.groupby('PM_num', as_index=False)['result'].mean()
df_spane_avg.rename(columns={'result': 'SPANE'}, inplace=True)

# --- 2. Read and parse baseline results ---
baseline_records = []
pattern_base = re.compile(r'N:\s*(\d+),\s*first fit:\s*([\d.]+),\s*balance fit:\s*([\d.]+)')
with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern_base.match(line.strip())
        if m:
            n   = int(m.group(1))
            ff  = float(m.group(2))
            bf  = float(m.group(3))
            baseline_records.append({'PM_num': n, 'ff': ff, 'bf': bf})

if not baseline_records:
    sys.exit(f"Error: no valid baseline records found in '{BASELINE_PATH}'.")

df_base = pd.DataFrame(baseline_records)

# --- 3. Merge and compute differences ---
df = pd.merge(df_spane_avg, df_base, on='PM_num', how='outer').sort_values('PM_num')

# Round SPANE, ff, bf to integers
df['SPANE'] = df['SPANE'].round().astype('Int64')
df['ff']    = df['ff'].round().astype('Int64')
df['bf']    = df['bf'].round().astype('Int64')

# Compute SPANE-bf and ff-bf
df['SPANE-bf'] = (df['SPANE'] - df['bf']).astype('Int64')
df['ff-bf']    = (df['ff']    - df['bf']).astype('Int64')

# Reorder columns
df = df[['PM_num', 'SPANE', 'bf', 'ff', 'SPANE-bf', 'ff-bf']]

# Save to CSV
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
print(f"Finished: results saved to '{OUTPUT_CSV_PATH}'.")
