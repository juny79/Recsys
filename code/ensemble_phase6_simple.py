"""
Phase 6: Simplification Strategy (ALS Only + Simple Boost)
가설: Phase 1이 너무 복잡할 수 있음. SASRec이 노이즈일 가능성
전략: ALS만 사용 + 간단한 Repeat Boost
예상: 0.1330 → 0.135~0.142 (+1.5~9%)
"""

import pandas as pd
import numpy as np
import argparse
import os

def load_simple_history(train_path):
    """간단한 히스토리: 카운트만"""
    print(f"Loading simple history from {train_path}...")
    
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id'])
    
    print(f"Total interactions: {len(df):,}")
    
    # 단순 카운트만
    print("  Computing interaction counts...")
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    print(f"✓ Processed {len(count_df):,} user-item pairs")
    
    return count_df

def apply_simple_boost(candidates_df, history_df):
    """간단한 Repeat Boost만 (Event/Recency 제외)"""
    print("Applying simple repeat boost...")
    
    # 1. 간단한 Repeat Boost
    print("  Calculating repeat boost...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.0
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.5
    
    # 2. 후보와 병합
    print("  Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'repeat_boost']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['repeat_boost'] = merged['repeat_boost'].fillna(1.0)
    
    # 3. 최종 점수
    merged['final_score'] = merged['final_score'] * merged['repeat_boost']
    
    # 4. Top-10 선택
    print("  Selecting Top-10 per user...")
    result_df = (merged
                 .sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id')
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Simplification")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase6_simple.csv", type=str)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 6: Simplification Strategy")
    print("Hypothesis: Simpler is better (ALS only + Repeat)")
    print("Target: 0.135~0.142 (+1.5~9%)")
    print("="*60)
    
    # 1. Load ALS only
    print(f"\n[1/3] Loading ALS output only...")
    als_df = pd.read_csv(args.als_output)
    print(f"  ✓ ALS: {als_df['user_id'].nunique():,} users")
    
    print(f"\n[2/3] Using ALS predictions (no ensemble)...")
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['final_score'] = 1.0 / als_df['rank']
    candidates_df = als_df[['user_id', 'item_id', 'final_score']]
    
    print(f"  ✓ {len(candidates_df):,} candidates")
    
    # 2. Load simple history
    print(f"\n[3/3] Loading simple history (count only)...")
    history_df = load_simple_history(args.train_data)
    
    # 3. Apply simple boost
    print(f"\nApplying simple optimization...")
    result_df = apply_simple_boost(candidates_df, history_df)
    
    # 4. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ Phase 6 (Simple) completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print()
    print("Key simplifications:")
    print("  • Model: ALS only (no SASRec)")
    print("  • Boost: Repeat only (no Event/Recency)")
    print("  • Values: 1.3/2.0/2.5 (count 1/2/3+)")
    print("="*60)

if __name__ == "__main__":
    main()
