import pandas as pd
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--output_dir", default="../data", type=str)
    args = parser.parse_args()
    
    train_path = os.path.join(args.data_dir, "train.parquet")
    print(f"Loading {train_path}...")
    df = pd.read_parquet(train_path)
    
    # 1. Item Features
    print("Generating Item Features...")
    item_stats = df.groupby('item_id').agg({
        'price': ['mean', 'min', 'max'],
        'event_type': 'count', # popularity
        'brand': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'category_code': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    })
    item_stats.columns = ['price_mean', 'price_min', 'price_max', 'pop_count', 'brand', 'category_code']
    item_stats = item_stats.reset_index()
    
    # View/Purchase Counts
    event_counts = df.pivot_table(index='item_id', columns='event_type', aggfunc='size', fill_value=0)
    for col in ['view', 'cart', 'purchase']:
        if col not in event_counts.columns:
            event_counts[col] = 0
            
    item_stats = pd.merge(item_stats, event_counts[['view', 'cart', 'purchase']], on='item_id', how='left')
    item_stats['conversion_rate'] = item_stats['purchase'] / (item_stats['view'] + 1)
    
    # Label Encode Brand/Category
    item_stats['brand'] = item_stats['brand'].astype('category').cat.codes
    item_stats['category_code'] = item_stats['category_code'].astype('category').cat.codes
    
    # 2. User Features
    print("Generating User Features...")
    user_stats = df.groupby('user_id').agg({
        'event_type': 'count', # activity
        'price': 'mean', # spending power
        'event_time': 'max' # last event
    })
    user_stats.columns = ['user_activity', 'user_avg_price', 'last_event_time']
    user_stats = user_stats.reset_index()
    
    
    # Save
    item_out = os.path.join(args.output_dir, "item_features.parquet")
    user_out = os.path.join(args.output_dir, "user_features.parquet")
    
    print(f"Saving to {item_out} and {user_out}...")
    item_stats.to_parquet(item_out)
    user_stats.to_parquet(user_out)
    print("Done.")

if __name__ == "__main__":
    main()
