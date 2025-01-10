import random
import string
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def print_best_result(results):
    if not results:
        print("No results available.")
        return
    best_result = max(results, key=lambda x: x['result'])
    print(f"Best result: {best_result['result']}")
    print(f"Best parameters: {best_result['params']}")
    print(f"Best trial ID: {best_result['trial_id']}")

def save_results(results, log_dir):
    results_list = list(results)
    sorted_results = sorted(results_list, key=lambda x: x['result'], reverse=True)
    with open(f'{log_dir}/all_results.json', 'w') as f:
        json.dump(sorted_results, f, indent=2)

def visualize_results(all_results, log_dir):
    # Process result
    df = pd.DataFrame(all_results)
    params_df = pd.json_normalize(df['params'])
    df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
    df['cluster_agg'] = df['cluster_agg'].astype(str)

    # Plot
    os.makedirs(f'{log_dir}/fig')
    for param in df.columns:
        if param not in ['trial_id', 'result']:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=param, y='result', data=df)
            plt.title(f'Impact of {param} on Result')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/fig/{param}_impact.png')
            plt.close()

def generate_trial_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=5))