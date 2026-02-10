import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_top100.csv", type=str)
    parser.add_argument("--user_features", default="../data/user_features.parquet", type=str)
    parser.add_argument("--item_features", default="../data/item_features.parquet", type=str)
    parser.add_argument("--output_dir", default="../data/", type=str)
    args = parser.parse_args()

    # 1. Load Pre-extracted Features
    print("Loading Features...")
    user_feats = pd.read_parquet(args.user_features)
    item_feats = pd.read_parquet(args.item_features)
    
    # 2. Load Model Outputs (Top-100)
    print("Loading Model Outputs...")
    als = pd.read_csv(args.als_output)
    sasrec = pd.read_csv(args.sasrec_output)
    
    # Assign ranks and RRF scores
    K = 60
    als['als_rank'] = als.groupby('user_id').cumcount() + 1
    als['als_rrf'] = 1 / (K + als['als_rank'])
    
    sasrec['sasrec_rank'] = sasrec.groupby('user_id').cumcount() + 1
    sasrec['sasrec_rrf'] = 1 / (K + sasrec['sasrec_rank'])
    
    # 3. Create Candidate Set (Union)
    print("Creating Candidate Set...")
    candidates = pd.merge(
        als[['user_id', 'item_id', 'als_rank', 'als_rrf']],
        sasrec[['user_id', 'item_id', 'sasrec_rank', 'sasrec_rrf']],
        on=['user_id', 'item_id'],
        how='outer'
    )
    
    # Fill missing ranks (if item not in Top-100, give it a high rank like 500)
    candidates['als_rank'] = candidates['als_rank'].fillna(500)
    candidates['sasrec_rank'] = candidates['sasrec_rank'].fillna(500)
    candidates['als_rrf'] = candidates['als_rrf'].fillna(0)
    candidates['sasrec_rrf'] = candidates['sasrec_rrf'].fillna(0)
    
    # 4. Merge with Features
    print("Merging Features...")
    candidates = pd.merge(candidates, user_feats, on='user_id', how='left')
    candidates = pd.merge(candidates, item_feats, on='item_id', how='left')
    
    # 5. Labeling (for Training Data Only)
    # We use the LAST interaction of each user as the target if possible
    # For inference, label will be dummy
    
    # Let's save the inference candidates first
    print(f"Saving {len(candidates)} Inference Candidates...")
    candidates.to_parquet(os.path.join(args.output_dir, "ltr_test_candidates.parquet"))
    
    # Generate Training Labels
    print("Generating Training Labels...")
    df = pd.read_parquet(args.train_file)
    # Get last interaction per user
    df = df.sort_values(['user_id', 'event_time'])
    last_inter = df.groupby('user_id').tail(1)[['user_id', 'item_id']]
    last_inter['label'] = 1
    
    # For training data, we only use users that have interactions in the candidate set
    train_data = pd.merge(candidates, last_inter, on=['user_id', 'item_id'], how='left')
    train_data['label'] = train_data['label'].fillna(0)
    
    # Undersampling Negatives to balance (Many 0s, few 1s)
    # Keep all positives, sample 10x negatives
    pos = train_data[train_data['label'] == 1]
    neg = train_data[train_data['label'] == 0]
    
    print(f"Positives: {len(pos)}, Negatives: {len(neg)}")
    
    if len(pos) > 0:
        neg_sample = neg.sample(n=min(len(neg), len(pos)*20), random_state=42)
        train_balanced = pd.concat([pos, neg_sample]).sample(frac=1.0, random_state=42)
        
        print(f"Saving {len(train_balanced)} Balanced Training Samples...")
        train_balanced.to_parquet(os.path.join(args.output_dir, "ltr_train_data.parquet"))
    else:
        print("Warning: No positive labels found in candidates. Check alignment.")

    print("Done.")

if __name__ == "__main__":
    main()
