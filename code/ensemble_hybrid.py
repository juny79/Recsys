import pandas as pd
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed.csv", type=str)
    parser.add_argument("--xgboost_output", default="../output/output_reranked.csv", type=str)
    parser.add_argument("--output_path", default="../output/output_ensemble_hybrid.csv", type=str)
    
    args = parser.parse_args()
    
    print(f"Loading data...")
    # Load Weights
    # Short History (< 5)
    w_short = {'als': 0.4, 'sasrec': 0.1, 'xgb': 0.5}
    # Long History (>= 5)
    w_long = {'als': 0.6, 'sasrec': 0.3, 'xgb': 0.1}
    
    print(f"Weights Short (<5): {w_short}")
    print(f"Weights Long (>=5): {w_long}")
    
    # Load Outputs
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgboost_output)
    
    # Load User History Count
    print("Calculating User History Lengths...")
    train_df = pd.read_parquet(args.train_file)
    user_counts = train_df.groupby('user_id').size()
    
    # Calculate Scores (1/Rank)
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['score_als'] = 1 / als_df['rank']
    
    sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
    sasrec_df['score_sasrec'] = 1 / sasrec_df['rank']
    
    xgb_df['rank'] = xgb_df.groupby('user_id').cumcount() + 1
    xgb_df['score_xgb'] = 1 / xgb_df['rank']
    
    # Merge All
    print("Merging outputs...")
    merged = pd.merge(als_df[['user_id', 'item_id', 'score_als']], 
                      sasrec_df[['user_id', 'item_id', 'score_sasrec']], 
                      on=['user_id', 'item_id'], how='outer')
    
    merged = pd.merge(merged, 
                      xgb_df[['user_id', 'item_id', 'score_xgb']],
                      on=['user_id', 'item_id'], how='outer')
    
    merged = merged.fillna(0)
    
    # Apply Segmented Weights
    print("Applying Segmented Weights...")
    
    # Map user history count to merged dataframe
    # optimize: create a dict or series for faster lookup
    # user_counts is a series index=user_id, value=count
    
    # We need to broadcast user counts to the merged dataframe (which has multiple rows per user)
    # merged['user_count'] = merged['user_id'].map(user_counts).fillna(0) # This might be slow if huge
    
    # Alternative: Join
    user_counts_df = user_counts.reset_index(name='count')
    merged = pd.merge(merged, user_counts_df, on='user_id', how='left')
    merged['count'] = merged['count'].fillna(0)
    
    # Vectorized calculation
    # Condition mask
    is_short = merged['count'] < 5
    
    # Calculate final score
    merged['final_score'] = 0.0
    
    # Short users
    merged.loc[is_short, 'final_score'] = (
        (w_short['als'] * merged.loc[is_short, 'score_als']) + 
        (w_short['sasrec'] * merged.loc[is_short, 'score_sasrec']) + 
        (w_short['xgb'] * merged.loc[is_short, 'score_xgb'])
    )
    
    # Long users
    merged.loc[~is_short, 'final_score'] = (
        (w_long['als'] * merged.loc[~is_short, 'score_als']) + 
        (w_long['sasrec'] * merged.loc[~is_short, 'score_sasrec']) + 
        (w_long['xgb'] * merged.loc[~is_short, 'score_xgb'])
    )
    
    # Select Top 10
    print("Selecting Top 10...")
    final_df = merged.sort_values(by=['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Hybrid Ensemble saved to {args.output_path}")
    print(f"Total predictions: {len(final_top10)}")

if __name__ == "__main__":
    main()
