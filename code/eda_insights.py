import pandas as pd
import numpy as np
import os

def analyze_repeat_behavior(df):
    print("\n[1] Analyzing Repeat Behavior (Loyalty)")
    # Group by user and item to see duplicates
    user_item_counts = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    repeat_rate = (user_item_counts['count'] > 1).mean()
    print(f"Overall Repeat Interaction Rate: {repeat_rate:.4f}")
    
    # Repeat rate by event type
    print("Repeat rate by event type:")
    for etype in ['view', 'cart', 'purchase']:
        sub = df[df['event_type'] == etype]
        counts = sub.groupby(['user_id', 'item_id']).size()
        rrate = (counts > 1).mean()
        print(f" - {etype}: {rrate:.4f}")

def analyze_category_transitions(df):
    print("\n[2] Analyzing Category Transitions")
    df = df.sort_values(['user_id', 'event_time'])
    df['prev_cat'] = df.groupby('user_id')['category_code'].shift(1)
    
    # Drop NAs and see top transitions
    transitions = df[df['prev_cat'].notna() & (df['prev_cat'] != df['category_code'])]
    trans_counts = transitions.groupby(['prev_cat', 'category_code']).size().reset_index(name='count')
    top_trans = trans_counts.sort_values('count', ascending=False).head(10)
    print("Top Category Transitions (Divergent Interest):")
    print(top_trans)
    
    # Stay rate in same category
    same_cat_rate = (df['prev_cat'] == df['category_code']).mean()
    print(f"Consecutive Same Category Rate: {same_cat_rate:.4f}")

def analyze_temporal_patterns(df):
    print("\n[3] Analyzing Temporal Patterns")
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour'] = df['event_time'].dt.hour
    
    # Activity by hour
    hour_counts = df['hour'].value_counts().sort_index()
    print("Activity by hour of day (UTC):")
    print(hour_counts)
    
    # Time since last interaction
    df['time_diff'] = df.groupby('user_id')['event_time'].diff().dt.total_seconds() / 60
    print("\nTime difference between interactions (minutes):")
    print(df['time_diff'].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

def main():
    train_path = "../data/train.parquet"
    print(f"Reading {train_path}...")
    df = pd.read_parquet(train_path)
    
    analyze_repeat_behavior(df)
    analyze_category_transitions(df)
    analyze_temporal_patterns(df)

if __name__ == "__main__":
    main()
