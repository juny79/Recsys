import pandas as pd
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="../output/output_decay01.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_advanced_ensemble.csv", type=str)
    # Threshold for short/long sequence
    parser.add_argument("--len_threshold", default=5, type=int)
    # Weight settings
    # Short history: Trust ALS completely (1.0)
    parser.add_argument("--short_als_w", default=1.0, type=float)
    parser.add_argument("--short_sasrec_w", default=0.0, type=float)
    # Long history: Mix ALS and SASRec (0.6 : 0.4)
    parser.add_argument("--long_als_w", default=0.6, type=float)
    parser.add_argument("--long_sasrec_w", default=0.4, type=float)
    
    args = parser.parse_args()

    print("Loading Data...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    train_df = pd.read_parquet(args.train_data)
    
    print("Calculating User History Lengths...")
    user_history_counts = train_df.groupby('user_id').size().reset_index(name='history_len')
    
    # Create a mapping for quick lookup
    user_len_map = dict(zip(user_history_counts['user_id'], user_history_counts['history_len']))
    
    print("Performing Rank-Based Ensemble with Segmentation...")
    
    # Compute Ranks & Scores
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['score'] = 1 / als_df['rank'] 
    
    sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
    sasrec_df['score'] = 1 / sasrec_df['rank']
    
    # Merge
    merged = pd.merge(als_df[['user_id', 'item_id', 'score']], 
                      sasrec_df[['user_id', 'item_id', 'score']], 
                      on=['user_id', 'item_id'], 
                      how='outer', 
                      suffixes=('_als', '_sasrec'))
    
    merged['score_als'] = merged['score_als'].fillna(0)
    merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
    
    # Apply Segmented Weights
    # We need to map history_len to each row in merged
    merged['history_len'] = merged['user_id'].map(user_len_map).fillna(0)
    
    def calculate_final_score(row):
        h_len = row['history_len']
        s_als = row['score_als']
        s_sasrec = row['score_sasrec']
        
        if h_len < args.len_threshold:
            # Short Sequence: ALS dominant
            return (args.short_als_w * s_als) + (args.short_sasrec_w * s_sasrec)
        else:
            # Long Sequence: Mixed
            return (args.long_als_w * s_als) + (args.long_sasrec_w * s_sasrec)

    merged['final_score'] = merged.apply(calculate_final_score, axis=1)
    
    # Sort and pick Top 10
    final_df = merged.sort_values(by=['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Statistics
    short_users = len(user_history_counts[user_history_counts['history_len'] < args.len_threshold])
    long_users = len(user_history_counts[user_history_counts['history_len'] >= args.len_threshold])
    print(f"User Segmentation Stats:")
    print(f" - Short Sequence Users (<{args.len_threshold}): {short_users} (ALS 100%)")
    print(f" - Long Sequence Users (>={args.len_threshold}): {long_users} (ALS 60% + SASRec 40%)")

    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Advanced Ensemble completed. Saved to {args.output_path}")

if __name__ == "__main__":
    main()
