import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/ltr_v5_train.parquet", type=str)
    parser.add_argument("--test_data", default="data/ltr_v5_test_candidates.parquet", type=str)
    parser.add_argument("--output_path", default="output/output_v5_final.csv", type=str)
    args = parser.parse_args()

    # 1. Load Data
    print("Loading V5 Training data...")
    train_df = pd.read_parquet(args.train_data)
    train_df = train_df.sort_values('user_id')
    
    # Features List (Based on Key Takeaways)
    features = [
        'als_score', 'sas_score', 'pop_score', 'v5_score',
        'last_hour', 'last_day',
        'is_repeat', 'brand_match',
        'user_price_mean', 'user_price_trend',
        'pop_count', 'pop_rank',
        'item_popularity', 'item_price'
    ]
    # Categorical
    cat_features = ['top_brand', 'item_brand', 'item_cat']
    features.extend(cat_features)
    
    # Fill NaN & Cast Types
    for f in features:
        if f in train_df.columns:
            if f in cat_features:
                train_df[f] = train_df[f].fillna('Unknown').astype(str)
            else:
                train_df[f] = train_df[f].fillna(0)
    
    print(f"Training on {len(features)} features...")
    
    # Pool
    train_pool = Pool(
        data=train_df[features],
        label=train_df['label'],
        group_id=train_df['user_id'],
        cat_features=cat_features
    )
    
    # 2. Train Model (Optimized for NDCG recovery)
    print("Initializing CatBoostRanker & Splitting Validation...")
    
    # Simple user-based split for validation monitoring
    unique_users = train_df['user_id'].unique()
    np.random.seed(42)
    val_users = np.random.choice(unique_users, size=int(len(unique_users)*0.1), replace=False)
    
    train_part = train_df[~train_df['user_id'].isin(val_users)]
    val_part = train_df[train_df['user_id'].isin(val_users)]
    
    train_pool = Pool(
        data=train_part[features],
        label=train_part['label'],
        group_id=train_part['user_id'],
        cat_features=cat_features
    )
    val_pool = Pool(
        data=val_part[features],
        label=val_part['label'],
        group_id=val_part['user_id'],
        cat_features=cat_features
    )
    
    import torch
    has_gpu = torch.cuda.is_available()
    device_type = 'GPU' if has_gpu else 'CPU'
    print(f"Using device: {device_type}")

    ranker_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'YetiRank',
        'eval_metric': 'NDCG:top=10;type=Base',
        'task_type': device_type,
        'random_seed': 42,
        'verbose': 100,
        'early_stopping_rounds': 50
    }
    
    if has_gpu:
        ranker_params['gpu_ram_part'] = 0.8
        print("VRAM limit set to 80%")

    ranker = CatBoostRanker(**ranker_params)
    
    print(f"Training CatBoostRanker (v5) on {device_type}...")
    ranker.fit(train_pool, eval_set=val_pool)
    
    # 3. Chunked Inference
    print("Inference on Expanded Candidates (Chunked)...")
    import pyarrow.parquet as pq
    table = pq.ParquetFile(args.test_data)
    header = True
    
    for i in tqdm(range(table.num_row_groups)):
        chunk_df = table.read_row_group(i).to_pandas()
        
        # Preprocess chunk
        for f in features:
            if f in chunk_df.columns:
                if f in cat_features:
                    chunk_df[f] = chunk_df[f].fillna('Unknown').astype(str)
                else:
                    chunk_df[f] = chunk_df[f].fillna(0)
            else:
                # Add missing features (could happen if test candidates lacks some train features)
                if f in cat_features:
                    chunk_df[f] = 'Unknown'
                else:
                    chunk_df[f] = 0
        
        # Predict
        chunk_df['score'] = ranker.predict(chunk_df[features])
        
        # Save temp
        chunk_df[['user_id', 'item_id', 'score']].to_csv("temp_v5_scores.csv", mode='a', index=False, header=header)
        header = False

    print("Finalizing Top 10 (Memory Efficient)...")
    # Read in chunks to avoid OOM for very large temp files
    chunk_size = 500000
    top10_list = []
    
    # We use a trick to keep only top 10 per user
    # First, get unique users to ensure we process them correctly if the file is large
    reader = pd.read_csv("temp_v5_scores.csv", chunksize=chunk_size)
    
    for chunk in reader:
        # Sort within chunk and keep a bit more than top 10 to be safe before global group
        top_candidates = chunk.sort_values(['user_id', 'score'], ascending=[True, False]).groupby('user_id').head(20)
        top10_list.append(top_candidates)
    
    # Combine the top candidates from all chunks
    combined_top = pd.concat(top10_list)
    final_top10 = combined_top.sort_values(['user_id', 'score'], ascending=[True, False]).groupby('user_id').head(10)[['user_id', 'item_id', 'score']]
    
    print(f"Saving to {args.output_path}...")
    final_top10.to_csv(args.output_path, index=False)
    
    if os.path.exists("temp_v5_scores.csv"):
        os.remove("temp_v5_scores.csv")
    print("Finish.")

if __name__ == "__main__":
    from tqdm import tqdm
    main()
