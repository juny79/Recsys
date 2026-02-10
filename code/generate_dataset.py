import pandas as pd
import numpy as np
import os
import argparse
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_top100.csv", type=str)
    parser.add_argument("--output_dir", default="../data/", type=str)
    args = parser.parse_args()

    # 1. Generate Training Data for Reranker
    print("Generating Ranker Training Data...")
    df = pd.read_parquet(args.train_file)
    
    # Positive samples: actual interactions
    # To reduce size/noise, maybe take last 5-10 items per user?
    # Or just use all 'purchase'/'view' events?
    # Let's use last 20 interactions per user to get enough coverage
    df['event_time'] = df['event_time'].fillna(0) # Handle potential NaNs
    df = df.sort_values(['user_id', 'event_time'])
    
    positives = df.groupby('user_id').tail(20)[['user_id', 'item_id']]
    positives['label'] = 1
    
    # Negative samples: random items
    all_items = df['item_id'].unique()
    users = positives['user_id'].unique()
    
    negatives = []
    # Sample 20 negatives per user
    # Optimized sampling
    n_neg = 20
    for u in tqdm(users):
        # fast random sample
        negs = np.random.choice(all_items, n_neg)
        for i in negs:
            negatives.append({'user_id': u, 'item_id': i, 'label': 0})
            
    neg_df = pd.DataFrame(negatives)
    
    train_ranker = pd.concat([positives, neg_df], ignore_index=True)
    train_ranker = train_ranker.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Train Ranker Size: {len(train_ranker)}")
    train_ranker.to_parquet(os.path.join(args.output_dir, "train_ranker.parquet"))
    
    # 2. Generate Test Candidates (Inference Target)
    print("Generating Test Candidates...")
    if os.path.exists(args.als_output):
        als = pd.read_csv(args.als_output)
    else:
        print(f"Warning: {args.als_output} not found. Using empty.")
        als = pd.DataFrame(columns=['user_id', 'item_id'])

    if os.path.exists(args.sasrec_output):
        sasrec = pd.read_csv(args.sasrec_output)
    else:
        print(f"Warning: {args.sasrec_output} not found. Using empty.")
        sasrec = pd.DataFrame(columns=['user_id', 'item_id'])
        
    # Union
    candidates = pd.concat([als[['user_id', 'item_id']], sasrec[['user_id', 'item_id']]]).drop_duplicates()
    
    print(f"Test Candidates Size: {len(candidates)}")
    candidates.to_parquet(os.path.join(args.output_dir, "test_ranker_candidates.parquet"))
    print("Done.")

if __name__ == "__main__":
    main()
