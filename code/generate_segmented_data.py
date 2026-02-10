import pandas as pd
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_top100.csv", type=str)
    parser.add_argument("--user_trajectory", default="../data/user_trajectory_v3.parquet", type=str)
    parser.add_argument("--item_stats", default="../data/item_stats_v3.parquet", type=str)
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--output_dir", default="../data/", type=str)
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Features...")
    user_traj = pd.read_parquet(args.user_trajectory)
    item_stats = pd.read_parquet(args.item_stats)
    
    # 2. Candidate Generation (Union)
    print("Loading Model Outputs & Merging Candidates...")
    als = pd.read_csv(args.als_output)
    sas = pd.read_csv(args.sasrec_output)
    
    K = 60
    als['als_rank'] = als.groupby('user_id').cumcount() + 1
    als['als_rrf'] = 1 / (K + als['als_rank'])
    
    sas['sas_rank'] = sas.groupby('user_id').cumcount() + 1
    sas['sas_rrf'] = 1 / (K + sas['rank' if 'rank' in sas.columns else 'sas_rank']) # Safety check
    
    candidates = pd.merge(
        als[['user_id', 'item_id', 'als_rank', 'als_rrf']],
        sas[['user_id', 'item_id', 'sas_rank', 'sas_rrf']],
        on=['user_id', 'item_id'],
        how='outer'
    )
    
    # Fill defaults
    candidates['als_rank'] = candidates['als_rank'].fillna(500)
    candidates['sas_rank'] = candidates['sas_rank'].fillna(500)
    candidates['als_rrf'] = candidates['als_rrf'].fillna(0)
    candidates['sas_rrf'] = candidates['sas_rrf'].fillna(0)
    
    # 3. Add Segmentation & Features
    print("Joining Features...")
    # Get interaction counts for segmentation
    df_train = pd.read_parquet(args.train_file)
    counts = df_train.groupby('user_id').size().reset_index(name='inter_count')
    
    candidates = pd.merge(candidates, counts, on='user_id', how='left')
    candidates['is_heavy'] = (candidates['inter_count'] >= 5).astype(int)
    
    candidates = pd.merge(candidates, user_traj, on='user_id', how='left')
    candidates = pd.merge(candidates, item_stats, on='item_id', how='left')
    
    # 4. Generate Training Labels (Last Interaction)
    print("Labeling for Training...")
    # Sort by time and pick last
    df_train = df_train.sort_values(['user_id', 'event_time'])
    last_inter = df_train.groupby('user_id').tail(1)[['user_id', 'item_id']]
    last_inter['label'] = 1
    
    train_data = pd.merge(candidates, last_inter, on=['user_id', 'item_id'], how='left')
    train_data['label'] = train_data['label'].fillna(0)
    
    # Stratified Sampling (Maintain 1:20 ratio for Lite/Heavy separately if possible)
    print("Saving Datasets...")
    # Inference Data (Full)
    candidates.to_parquet(os.path.join(args.output_dir, "ltr_v3_test_candidates.parquet"))
    
    # Balanced Training Data
    pos = train_data[train_data['label'] == 1]
    neg = train_data[train_data['label'] == 0]
    
    if len(pos) > 0:
        neg_sample = neg.sample(n=min(len(neg), len(pos)*20), random_state=42)
        train_final = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=42)
        train_final.to_parquet(os.path.join(args.output_dir, "ltr_v3_train_data.parquet"))
        print(f"Saved {len(train_final)} training rows.")
    
    print("Done.")

if __name__ == "__main__":
    main()
