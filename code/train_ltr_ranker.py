import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../data/ltr_train_data.parquet", type=str)
    parser.add_argument("--test_data", default="../data/ltr_test_candidates.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_ltr_final.csv", type=str)
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading Training data...")
    train_df = pd.read_parquet(args.train_data)
    
    # Sort by user_id for group-based ranking
    train_df = train_df.sort_values('user_id')
    
    # Features
    features = [
        'als_rank', 'als_rrf', 'sasrec_rank', 'sasrec_rrf',
        'user_activity', 'user_avg_price',
        'price_mean', 'price_min', 'price_max', 'pop_count',
        'view', 'cart', 'purchase', 'conversion_rate',
        'brand', 'category_code'
    ]
    
    X_train = train_df[features]
    y_train = train_df['label']
    groups = train_df.groupby('user_id').size().values
    
    # 2. Train Ranker
    print("Training XGBRanker...")
    ranker = xgb.XGBRanker(
        objective='rank:pairwise',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist',
        device='cuda', # Forced to cuda, most likely available
        random_state=42
    )
    
    ranker.fit(X_train, y_train, group=groups, verbose=True)
    
    # 3. Inference (Chunked with pyarrow to save memory)
    print("Inference on 67M candidates (Chunked)...")
    import pyarrow.parquet as pq
    
    table = pq.ParquetFile(args.test_data)
    header = True
    
    for i in range(table.num_row_groups):
        chunk_df = table.read_row_group(i).to_pandas()
        
        # Predict
        chunk_df['score'] = ranker.predict(chunk_df[features])
        
        # Save results iteratively
        chunk_df[['user_id', 'item_id', 'score']].to_csv("temp_scores.csv", mode='a', index=False, header=header)
        header = False
        if i % 10 == 0:
            print(f"Processed row group {i}/{table.num_row_groups}...")

    print("Selecting Final Top 10 per user from temp scores...")
    # Read temp scores in chunks for final top-k to avoid memory spikes
    # Since we need to group by user, we can't easily avoid full load unless sorted.
    # But temp_df is only 3 columns, should be fine.
    temp_df = pd.read_csv("temp_scores.csv")
    final_df = temp_df.sort_values(['user_id', 'score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)[['user_id', 'item_id']]
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    print(f"Saving to {args.output_path}...")
    final_top10.to_csv(args.output_path, index=False)
    
    # Cleanup
    if os.path.exists("temp_scores.csv"):
        os.remove("temp_scores.csv")
    print("Done.")

if __name__ == "__main__":
    main()
