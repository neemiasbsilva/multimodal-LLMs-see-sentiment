import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Define paths for GPT-4 OpenAI
input_folder = "/mnt/raid5/neemias/PerceptSent-LLM-approach/data/gpt4-openai-classify/"
response_folder = "/mnt/raid5/neemias/PerceptSent-LLM-approach/data/gpt4-openai-only/"
output_folder = "/mnt/raid5/neemias/PerceptSent-LLM-approach/data/dataset_gpt4_ids"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# List all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
print(f"Found {len(csv_files)} CSV files in input folder")
response_files = [f for f in os.listdir(response_folder) if f.endswith(".csv")]
print(f"Found {len(response_files)} CSV files in response folder")

for csv_file in csv_files:
    print(f"===Processing input file: {csv_file}===")
    df = pd.read_csv(os.path.join(input_folder, csv_file))
    print(f"Input file has {len(df)} rows")
    
    # Extract alpha and p values from filename
    # Example: percept_dataset_alpha3_p2neg.csv -> alpha3, p2
    filename_parts = csv_file.replace('percept_dataset_', '').replace('.csv', '')
    
    # Handle different naming patterns
    if 'alpha' in filename_parts and 'p' in filename_parts:
        alpha_part = filename_parts.split('_')[0]  # alpha3, alpha4, alpha5
        p_part = filename_parts.split('_')[1]     # p2neg, p2plus, p3, p5
        
        # Clean up p_part to match response file naming
        if 'neg' in p_part:
            p_part = p_part.replace('neg', '')
        elif 'plus' in p_part:
            p_part = p_part.replace('plus', '')
        
        # Construct response filename
        response_filename = f"{alpha_part}_{p_part}.csv"
        print(f"Looking for response file: {response_filename}")
        
        # Check if response file exists
        if response_filename in response_files:
            df_response = pd.read_csv(os.path.join(response_folder, response_filename))
            print(f"Response file has {len(df_response)} rows")
            
            # Filter input data to only include IDs present in response data
            df = df[df["id"].isin(df_response["id"])]
            print(f"After filtering: {len(df)} rows")
            
            # Sort both dataframes by ID
            df = df.sort_values(by='id').reset_index(drop=True)
            df_response = df_response.sort_values(by='id').reset_index(drop=True)
            
            # Perform 5-fold cross-validation
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
                train_df = df.iloc[train_idx][['id', 'text', 'sentiment']]
                val_df = df.iloc[val_idx][['id', 'text', 'sentiment']]
                
                # Create output filenames
                base_name = os.path.splitext(csv_file)[0]
                train_filename = f"{base_name}_fold{fold}_train.csv"
                val_filename = f"{base_name}_fold{fold}_val.csv"
                
                # Save train and validation sets
                train_df.to_csv(os.path.join(output_folder, train_filename), index=False)
                val_df.to_csv(os.path.join(output_folder, val_filename), index=False)
                
                print(f"  Fold {fold}: Train={len(train_df)}, Val={len(val_df)}")
        else:
            print(f"Response file {response_filename} not found, skipping...")
    else:
        print(f"Could not parse filename {csv_file}, skipping...")

print("GPT-4 OpenAI experiment completed!")

