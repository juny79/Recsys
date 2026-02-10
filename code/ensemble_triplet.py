import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed.csv", type=str)
    parser.add_argument("--xgboost_output", default="../output/output_reranked.csv", type=str)
    parser.add_argument("--output_path", default="../output/output_ensemble_triplet.csv", type=str)
    
    # Weights
    parser.add_argument("--w_als", default=0.6, type=float)
    parser.add_argument("--w_sasrec", default=0.25, type=float)
    parser.add_argument("--w_xgb", default=0.15, type=float)
    
    args = parser.parse_args()
    
    # RRF Constant
    K = 60
    
    print(f"Loading outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgboost_output)
    
    # Assign Scores based on RRF (1 / (K + rank))
    # ALS
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['score_als'] = 1 / (K + als_df['rank'])
    
    # SASRec
    sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
    sasrec_df['score_sasrec'] = 1 / (K + sasrec_df['rank'])
    
    # XGBoost
    xgb_df['rank'] = xgb_df.groupby('user_id').cumcount() + 1
    xgb_df['score_xgb'] = 1 / (K + xgb_df['rank'])
    
    print("Merging...")
    # Merge ALS + SASRec
    merged = pd.merge(als_df[['user_id', 'item_id', 'score_als']], 
                      sasrec_df[['user_id', 'item_id', 'score_sasrec']], 
                      on=['user_id', 'item_id'], 
                      how='outer')
    
    # Merge + XGBoost
    merged = pd.merge(merged, 
                      xgb_df[['user_id', 'item_id', 'score_xgb']],
                      on=['user_id', 'item_id'],
                      how='outer')
    
    # Fill NaN
    merged['score_als'] = merged['score_als'].fillna(0)
    merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
    merged['score_xgb'] = merged['score_xgb'].fillna(0)
    
    # Calculate Final Score
    print(f"Applying Weights: ALS={args.w_als}, SASRec={args.w_sasrec}, XGB={args.w_xgb}")
    merged['final_score'] = (
        (args.w_als * merged['score_als']) + 
        (args.w_sasrec * merged['score_sasrec']) + 
        (args.w_xgb * merged['score_xgb'])
    )
    
    # Select Top 10
    final_df = merged.sort_values(by=['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Triplet Ensemble saved to {args.output_path}")

if __name__ == "__main__":
    main()
