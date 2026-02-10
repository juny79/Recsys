import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--output_dir", default="../data/", type=str)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.train_file)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Sort for trajectory calculation
    df = df.sort_values(['user_id', 'event_time'])

    print("Extracting Trajectory Features...")
    
    # 1. Price Trend (Is the user looking at more expensive items?)
    # Calculate price change compared to previous interaction
    df['price_diff'] = df.groupby('user_id')['price'].diff()
    
    user_trajectory = df.groupby('user_id').agg({
        'price': ['mean', 'std', 'last'],
        'price_diff': 'mean',
        'brand': 'nunique',
        'category_code': 'nunique',
        'event_time': lambda x: (x.max() - x.min()).total_seconds() / 3600 # Duration in hours
    })
    
    user_trajectory.columns = [
        'user_price_mean', 'user_price_std', 'user_last_price',
        'user_price_trend', 'user_brand_count', 'user_cat_count',
        'user_duration_hours'
    ]
    user_trajectory['user_activity_density'] = df.groupby('user_id').size() / (user_trajectory['user_duration_hours'] + 1)
    
    # 2. Brand Frequency / Loyalty
    # Most frequent brand per user
    print("Calculating Brand Loyalty...")
    user_brand_stats = df.groupby(['user_id', 'brand']).size().reset_index(name='brand_freq')
    user_brand_stats = user_brand_stats.sort_values(['user_id', 'brand_freq'], ascending=[True, False])
    user_top_brand = user_brand_stats.groupby('user_id').head(1).rename(columns={'brand': 'top_brand', 'brand_freq': 'top_brand_freq'})
    
    user_trajectory = pd.merge(user_trajectory.reset_index(), user_top_brand[['user_id', 'top_brand', 'top_brand_freq']], on='user_id', how='left')
    
    # 3. Item Features (Enhanced)
    print("Enhancing Item Features...")
    item_stats = df.groupby('item_id').agg({
        'price': 'mean',
        'event_type': 'count',
        'brand': 'first',
        'category_code': 'first'
    })
    item_stats.columns = ['item_price', 'item_popularity', 'item_brand', 'item_cat']
    
    # Save
    user_out = os.path.join(args.output_dir, "user_trajectory_v3.parquet")
    item_out = os.path.join(args.output_dir, "item_stats_v3.parquet")
    
    print(f"Saving to {user_out} and {item_out}...")
    user_trajectory.to_parquet(user_out)
    item_stats.reset_index().to_parquet(item_out)
    print("Done.")

if __name__ == "__main__":
    main()
