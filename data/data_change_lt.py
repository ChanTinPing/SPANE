''' Convert the original Huawei-East-1 dataset to a dataset with length time (lt) and arrival time (at) '''

import pandas as pd
from tqdm import tqdm

def f():
    # Read the CSV file
    df = pd.read_csv('Huawei-East-1.csv')

    # Create a new DataFrame to store the transformed data
    new_data = []

    # Iterate over unique VM IDs in the original data
    for vmid in tqdm(df['vmid'].unique()):
        # Get the creation and deletion records for the current VM ID
        creation_record = df[(df['vmid'] == vmid) & (df['type'] == 0)]
        deletion_record = df[(df['vmid'] == vmid) & (df['type'] == 1)]
        
        # If there is a deletion record, calculate the length time (lt)
        if not deletion_record.empty:
            cpu = creation_record['cpu'].values[0]
            mem = creation_record['memory'].values[0]
            at = creation_record['time'].values[0]
            lt = deletion_record['time'].values[0] - at
            
            # Add the transformed data to the new dataset
            new_data.append([vmid, cpu, mem, at, lt])

    # Create a new DataFrame
    new_df = pd.DataFrame(new_data, columns=['vmid', 'cpu', 'memory', 'at', 'lt'])

    # Check if VM IDs are consecutive
    expected_vmid = list(range(len(new_df)))
    actual_vmid = new_df['vmid'].tolist()

    if actual_vmid != expected_vmid:
        print("VMID is not consecutive. Reordering VMIDs to be consecutive.")
        new_df = new_df.sort_values(by='at').reset_index(drop=True)
        new_df['vmid'] = range(len(new_df))
    else:
        print("VMID is consecutive.")

    # Check if arrival times (at) are in non-decreasing order
    if all(new_df['at'].iloc[i] <= new_df['at'].iloc[i+1] for i in range(len(new_df)-1)):
        print("Arrival times (at) are in non-decreasing order.")
    else:
        print("Arrival times (at) are not in non-decreasing order. Please check the data.")

    # Save the corrected dataset
    new_df = new_df.rename(columns={'memory': 'mem'})
    new_df.to_csv('Huawei-East-1-lt.csv', index=False)
    print("Data validation complete. The corrected dataset has been saved as 'Huawei-East-1-lt.csv'.")

if __name__ == '__main__':
    f()
