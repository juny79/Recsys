import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../data/ltr_v3_train_data.parquet", type=str)
    parser.add_argument("--test_data", default="../data/ltr_v3_test_candidates.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_catboost_final.csv", type=str)
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Training data...")
    train_df = pd.read_parquet(args.train_data)
    
    # Sort by user_id for CatBoost Pool (required for rankings)
    train_df = train_df.sort_values('user_id')
    
    # Features
    features = [
        'als_rank', 'als_rrf', 'sas_rank', 'sas_rrf',
        'is_heavy', 'user_price_mean', 'user_price_std', 'user_last_price',
        'user_price_trend', 'user_brand_count', 'user_cat_count',
        'user_activity_density', 'top_brand_freq',
        'item_price', 'item_popularity'
    ]
    # Categorical Features
    cat_features = ['top_brand', 'item_brand', 'item_cat']
    # Add categorical to features list if not already there
    features.extend(cat_features)
    
    # Fill NaN
    for f in features:
        if f in train_df.columns:
            if train_df[f].dtype == 'object' or train_df[f].dtype.name == 'category':
                train_df[f] = train_df[f].fillna('Unknown').astype(str)
            else:
                train_df[f] = train_df[f].fillna(0)
    
    # Pool creation
    train_pool = Pool(
        data=train_df[features],
        label=train_df['label'],
        group_id=train_df['user_id'],
        cat_features=cat_features
    )
    
    # 2. Train CatBoostRanker
    print("Training CatBoostRanker (TaskType=GPU)...")
    ranker = CatBoostRanker(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function='YetiRank', # Optimizes for NDCG
        task_type='GPU', # Use GPU
        random_seed=42,
        verbose=100
    )
    
    ranker.fit(train_pool)
    
    # 3. Inference (Chunked with pyarrow to save memory)
    print("Inference on 67M candidates (Chunked)...")
    import pyarrow.parquet as pq
    
    table = pq.ParquetFile(args.test_data)
    header = True
    
    for i in range(table.num_row_groups):
        chunk_df = table.read_row_group(i).to_pandas()
        
        # Preprocess chunk (Fill NaN, cast types)
        for f in features:
            if f in chunk_df.columns:
                if f in cat_features:
                    chunk_df[f] = chunk_df[f].fillna('Unknown').astype(str)
                else:
                    chunk_df[f] = chunk_df[f].fillna(0)
        
        # Predict
        chunk_df['score'] = ranker.predict(chunk_df[features])
        
        # Save results iteratively
        chunk_df[['user_id', 'item_id', 'score']].to_csv("temp_scores_cat.csv", mode='a', index=False, header=header)
        header = False
        if i % 10 == 0:
            print(f"Processed row group {i}/{table.num_row_groups}...")

    print("Selecting Final Top 10 per user from temp scores...")
    temp_df = pd.read_csv("temp_scores_cat.csv")
    final_df = temp_df.sort_values(['user_id', 'score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)[['user_id', 'item_id']]
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    print(f"Saving to {args.output_path}...")
    final_top10.to_csv(args.output_path, index=False)
    
    # Cleanup
    if os.path.exists("temp_scores_cat.csv"):
        os.remove("temp_scores_cat.csv")
    print("Done.")

if __name__ == "__main__":
    main()
