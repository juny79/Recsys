import pandas as pd
import numpy as np
import os
from collections import Counter

def main():
    # 1. Load best ensemble output (v5 version)
    base_file = "output/output_v5_final.csv"
    train_file = "data/train.parquet"
    item_stats_file = "data/item_stats_v3.parquet"
    output_file = "output/output_eda_boosted.csv"
    
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found!")
        return

    print(f"Loading base recommendations: {base_file}")
    rec_df = pd.read_csv(base_file)
    
    print("Loading user history & item stats...")
    # 1-1. Load only necessary columns to save memory
    train_df = pd.read_parquet(train_file, columns=['user_id', 'item_id', 'event_time', 'category_code'])
    
    # 1-2. Downcast numeric types
    train_df['user_id'] = train_df['user_id'].astype('int32')
    train_df['item_id'] = train_df['item_id'].astype('int32')
    
    item_stats = pd.read_parquet(item_stats_file, columns=['item_id', 'item_cat'])
    item_stats['item_id'] = item_stats['item_id'].astype('int32')
    
    # Get user history items & last category
    print("Processing user context...")
    train_df = train_df.sort_values(['user_id', 'event_time'])
    
    # Efficiently get last 50 items and last category per user
    user_context = train_df.groupby('user_id').agg({
        'item_id': lambda x: list(x)[-50:],
        'category_code': 'last'
    }).reset_index().rename(columns={'category_code': 'last_cat', 'item_id': 'history_items'})
    
    # Explicitly clear memory
    import gc
    del train_df
    gc.collect()
    
    # 2. Boosting Logic
    print("Applying Boosting Rules...")
    # Add item categories to recommendations
    rec_df['user_id'] = rec_df['user_id'].astype('int32')
    rec_df['item_id'] = rec_df['item_id'].astype('int32')
    
    merged = rec_df.merge(item_stats, on='item_id', how='left')
    merged = merged.merge(user_context, on='user_id', how='left')
    
    # Explicitly clear memory
    del rec_df, item_stats, user_context
    gc.collect()

    # Base Score from Model
    merged['rank'] = merged.groupby('user_id').cumcount() + 1
    if 'score' in merged.columns:
        print("Using model confidence scores as base...")
        min_s, max_s = merged['score'].min(), merged['score'].max()
        if max_s > min_s:
            merged['base_score'] = ((merged['score'] - min_s) / (max_s - min_s) + 0.1).astype('float32')
        else:
            merged['base_score'] = 1.0
    else:
        merged['base_score'] = (1 / (10 + merged['rank'])).astype('float32')
    
    # Rule 1: Loyalty Frequency Boost
    print("Computing Loyalty Frequency Boost...")
    
    # Optimized repeat count calculation using set lookup and list operations
    def calculate_repeat_boost(row):
        history = row['history_items']
        if not isinstance(history, list): return 0
        return history.count(row['item_id'])

    # Instead of dictionary which takes much memory, 
    # we can use the existing columns or a more memory-efficient way
    merged['repeat_count'] = merged.apply(calculate_repeat_boost, axis=1).astype('int16')
    
    # Rule 2: Category Alignment Boost
    print("Computing Category Boost...")
    merged['cat_match'] = (merged['item_cat'] == merged['last_cat']).astype('int8')
    
    # Final Score Calculation (Multiplicative)
    print("Calculating final scores...")
    merged['final_score'] = (merged['base_score'] * (1 + (merged['repeat_count'] > 0) * 0.5 + 
                                                   np.log1p(merged['repeat_count']) * 0.2 + 
                                                   merged['cat_match'] * 0.3)).astype('float32')
    
    # 3. Re-rank and Save
    print("Re-ranking and selecting Top 10...")
    final_df = merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
    
    # Keep minimal information
    final_top10 = final_df.groupby('user_id').head(10)[['user_id', 'item_id']]
    
    print(f"Saving to {output_file}...")
    final_top10.to_csv(output_file, index=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
