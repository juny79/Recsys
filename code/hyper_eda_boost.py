import pandas as pd
import numpy as np
import os

def main():
    # 1. Paths & Setup
    base_file = "../output/output_ensemble_triplet_27.csv"
    train_file = "../data/train.parquet"
    item_stats_file = "../data/item_stats_v3.parquet"
    output_file = "../output/output_eda_hyper_boosted.csv"
    
    if not os.path.exists(base_file):
        print(f"Error: {base_file} not found!")
        return

    print(f"Loading base recommendations: {base_file}")
    rec_df = pd.read_csv(base_file)
    
    print("Loading user history & item stats...")
    train_df = pd.read_parquet(train_file)
    item_stats = pd.read_parquet(item_stats_file)
    
    # Pre-calculate item metadata map for speed
    item_map = item_stats.set_index('item_id')[['item_brand', 'item_cat', 'item_price']].to_dict('index')
    
    # 2. Extract Deep User Context
    print("Gathering deep user context (Recency, Brand, Price)...")
    train_df = train_df.sort_values(['user_id', 'event_time'])
    
    # Grouping to get time-weighted history
    user_context = train_df.groupby('user_id').agg({
        'item_id': lambda x: list(x)[-20:], # Last 20 items
        'brand': lambda x: x.value_counts().index[0] if not x.mode().empty else 'Unknown', # Top brand
        'category_code': 'last', # Last seen category
        'price': ['mean', 'std']
    }).reset_index()
    
    user_context.columns = ['user_id', 'history_items', 'top_brand', 'last_cat', 'price_mean', 'price_std']
    
    # 3. Apply Hyper-Adjusted Boosting
    print("Applying Multi-dimensional Boosting...")
    merged = pd.merge(rec_df, user_context, on='user_id', how='left')
    
    # Base Rank/Score
    merged['rank'] = merged.groupby('user_id').cumcount() + 1
    merged['score'] = 1 / (10 + merged['rank'])
    
    def calculate_hyper_boost(row):
        score = row['score']
        item_id = row['item_id']
        meta = item_map.get(item_id, {})
        
        # Rule 1: Recency-Weighted Loyalty Boost
        # Items at the end of the history list (more recent) get higher boost
        if isinstance(row['history_items'], list) and item_id in row['history_items']:
            pos = row['history_items'].index(item_id)
            # Boost factor ranges from 1.2 (oldest in history) to 1.6 (most recent)
            recency_factor = 1.2 + (pos / len(row['history_items'])) * 0.4
            score *= recency_factor
            
        # Rule 2: Brand Consistency Boost
        if meta.get('item_brand') == row['top_brand'] and row['top_brand'] != 'Unknown':
            score *= 1.15 # 15% boost for user's favorite brand
            
        # Rule 3: Category Match
        if meta.get('item_cat') == row['last_cat']:
            score *= 1.25 # 25% boost for last seen category
            
        # Rule 4: Price Guard
        # If item is > 3x mean or < 0.3x mean, apply a slight penalty
        user_mean = row['price_mean']
        item_price = meta.get('item_price', 0)
        if user_mean > 0 and item_price > 0:
            if item_price > user_mean * 3 or item_price < user_mean * 0.3:
                score *= 0.9 # 10% penalty for price outliers
                
        return score

    # Apply boosting (using vectorized Ops where possible for speed, but deep comparison might need apply)
    print("Executing final score calculation...")
    merged['final_score'] = merged.apply(calculate_hyper_boost, axis=1)
    
    # 4. Re-rank and Save
    print("Re-ranking and selecting Top 10...")
    final_df = merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)[['user_id', 'item_id']]
    
    print(f"Saving to {output_file}...")
    final_top10.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
