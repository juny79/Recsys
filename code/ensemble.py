import pandas as pd
import argparse
import os
import numpy as np

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="../output/output_als.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed.csv", type=str)
    parser.add_argument("--output_path", default="../output/output_ensemble.csv", type=str)
    # Weights for ensemble: ALS is baseline (0.7), SASRec is supplementary (0.3)
    # This can be tuned.
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    parser.add_argument("--k", default=60, type=int, help="RRF constant K")
    args = parser.parse_args()

    print(f"Loading ALS output from {args.als_output}...")
    if not os.path.exists(args.als_output):
        print(f"Error: {args.als_output} not found.")
        return
    als_df = pd.read_csv(args.als_output)
    
    print(f"Loading SASRec output from {args.sasrec_output}...")
    if not os.path.exists(args.sasrec_output):
        print(f"Error: {args.sasrec_output} not found.")
        return
    sasrec_df = pd.read_csv(args.sasrec_output)

    # Calculate scores if not present (assuming rank-based if no scores)
    # Since the output is just user_id, item_id pairs (Top-10), we might not have scores.
    # If we don't have scores, we can use Rank-based ensemble (Reciprocal Rank).
    
    print(f"Performing Rank-Based Ensemble (RRF with K={args.k})...")
    
    # Assign ranks (1 to 10)
    # ALS
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['score'] = 1 / (args.k + als_df['rank']) # RRF Formula
    
    # SASRec
    sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
    sasrec_df['score'] = 1 / (args.k + sasrec_df['rank']) # RRF Formula
    
    # Merge
    # We want to combine scores for (user, item) pairs
    merged = pd.merge(als_df[['user_id', 'item_id', 'score']], 
                      sasrec_df[['user_id', 'item_id', 'score']], 
                      on=['user_id', 'item_id'], 
                      how='outer', 
                      suffixes=('_als', '_sasrec'))
    
    merged['score_als'] = merged['score_als'].fillna(0)
    merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
    
    # Weighted Sum
    # Note: For RRF, weights can still be applied if we trust one model more
    # Default is simple sum (1.0 : 1.0) for pure RRF, but we keep weights for flexibility
    merged['final_score'] = (args.w_als * merged['score_als']) + (args.w_sasrec * merged['score_sasrec'])
    
    # Sort and pick Text 10
    final_df = merged.sort_values(by=['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Ensemble completed. Saved to {args.output_path}")
    print(f"Total predictions: {len(final_top10)}")

if __name__ == "__main__":
    main()
