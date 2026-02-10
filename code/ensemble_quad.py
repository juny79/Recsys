import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="../output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed.csv", type=str)
    parser.add_argument("--ease_output", default="../output/output_ease.csv", type=str)
    parser.add_argument("--xgboost_output", default="../output/output_reranked.csv", type=str)
    parser.add_argument("--output_path", default="../output/output_ensemble_quad_v2.csv", type=str)
    
    # Weights (Total 1.0)
    # ALS: Proven baseline (0.4)
    # EASE: New strong linear model (0.3)
    # SASRec: Sequential signals (0.2)
    # XGB: Popularity/Content supplement (0.1)
    parser.add_argument("--w_als", default=0.4, type=float)
    parser.add_argument("--w_ease", default=0.3, type=float)
    parser.add_argument("--w_sasrec", default=0.2, type=float)
    parser.add_argument("--w_xgb", default=0.1, type=float)
    
    args = parser.parse_args()
    
    print(f"Loading outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    
    # Check if EASE exists (might be running)
    if not os.path.exists(args.ease_output):
        print(f"Error: {args.ease_output} not found. Please wait for EASE training to finish.")
        return
    ease_df = pd.read_csv(args.ease_output)
    
    xgb_df = pd.read_csv(args.xgboost_output)
    
    print("Calculating Rank Scores (RRF K=60)...")
    
    # RRF Constant
    K = 60
    
    # helper
    def get_score_df(df, name):
        df = df.copy()
        df['rank'] = df.groupby('user_id').cumcount() + 1
        df[f'score_{name}'] = 1 / (K + df['rank'])
        return df[['user_id', 'item_id', f'score_{name}']]

    als_score = get_score_df(als_df, 'als')
    sasrec_score = get_score_df(sasrec_df, 'sasrec')
    ease_score = get_score_df(ease_df, 'ease')
    xgb_score = get_score_df(xgb_df, 'xgb')
    
    print("Merging...")
    # Merge iteratively
    merged = pd.merge(als_score, sasrec_score, on=['user_id', 'item_id'], how='outer')
    merged = pd.merge(merged, ease_score, on=['user_id', 'item_id'], how='outer')
    merged = pd.merge(merged, xgb_score, on=['user_id', 'item_id'], how='outer')
    
    # Fill NaN
    merged = merged.fillna(0)
    
    # Weighted Sum of RRF Scores
    print(f"Applying Weights: ALS={args.w_als}, EASE={args.w_ease}, SASRec={args.w_sasrec}, XGB={args.w_xgb}")
    merged['final_score'] = (
        (args.w_als * merged['score_als']) + 
        (args.w_ease * merged['score_ease']) + 
        (args.w_sasrec * merged['score_sasrec']) + 
        (args.w_xgb * merged['score_xgb'])
    )
    
    # Select Top 10
    print("Selecting Top 10...")
    final_df = merged.sort_values(by=['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Quad Ensemble saved to {args.output_path}")

if __name__ == "__main__":
    main()
