"""
경량 버전: 핵심 Post-Processing만 적용
- 빠른 실행을 위해 전체 히스토리 로딩 생략
- Repeat Boost만 적용 (가장 효과적인 전략)
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

def load_user_recent_items(train_path):
    """사용자의 최근 상호작용 아이템만 빠르게 로딩"""
    print(f"Loading recent interactions from {train_path}...")
    
    # 필요한 컬럼만
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type'])
    
    print(f"Total interactions: {len(df):,}")
    
    # 유저별 아이템 상호작용 횟수 (가장 간단한 집계)
    user_item_counts = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    # 딕셔너리로 변환
    user_history = defaultdict(dict)
    for _, row in tqdm(user_item_counts.iterrows(), total=len(user_item_counts), desc="Building history"):
        user_history[row['user_id']][row['item_id']] = row['count']
    
    print(f"✓ Loaded history for {len(user_history):,} users")
    
    return user_history

def apply_repeat_boost(candidates_df, user_history, args):
    """Repeat Boost 적용 (벡터화 - 초고속!)"""
    print("Applying Repeat Boost (vectorized for speed)...")
    
    # 1. 히스토리를 DataFrame으로 변환
    print("  Converting history to DataFrame...")
    hist_data = []
    for user_id, items in tqdm(user_history.items(), desc="  Converting"):
        for item_id, count in items.items():
            hist_data.append({'user_id': user_id, 'item_id': item_id, 'count': count})
    
    hist_df = pd.DataFrame(hist_data)
    print(f"  ✓ History: {len(hist_df):,} user-item pairs")
    
    # 2. Boost 계산 (벡터화)
    hist_df['boost'] = 1.0
    hist_df.loc[hist_df['count'] == 1, 'boost'] = 1.3
    hist_df.loc[hist_df['count'] == 2, 'boost'] = 2.0
    hist_df.loc[hist_df['count'] >= 3, 'boost'] = 2.5
    
    # 3. 후보와 머지
    print(f"  Merging {len(candidates_df):,} candidates with history...")
    merged = candidates_df.merge(
        hist_df[['user_id', 'item_id', 'boost']], 
        on=['user_id', 'item_id'], 
        how='left'
    )
    merged['boost'] = merged['boost'].fillna(1.0)
    
    # 4. 최종 점수 계산 (벡터화)
    print("  Calculating final scores...")
    merged['final_score'] = merged['final_score'] * merged['boost']
    
    # 5. Top-10 선택 (벡터화)
    print("  Selecting Top-10 per user...")
    result_df = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id', sort=False)
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Fast nDCG@10 Optimization (Repeat Boost Only)")
    
    # Input files
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    
    # Output
    parser.add_argument("--output_path", default="../output/output_fast_optimized.csv", type=str)
    
    # Ensemble weights
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Fast nDCG@10 Optimization (Repeat Boost)")
    print("="*60)
    
    # 1. Load model outputs
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    print(f"  ✓ ALS: {als_df['user_id'].nunique():,} users")
    
    if os.path.exists(args.sasrec_output):
        sasrec_df = pd.read_csv(args.sasrec_output)
        print(f"  ✓ SASRec: {sasrec_df['user_id'].nunique():,} users")
        
        # Base ensemble
        print(f"\n[2/4] Creating base ensemble...")
        als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
        als_df['score'] = 1.0 / als_df['rank']
        
        sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
        sasrec_df['score'] = 1.0 / sasrec_df['rank']
        
        merged = pd.merge(
            als_df[['user_id', 'item_id', 'score']], 
            sasrec_df[['user_id', 'item_id', 'score']], 
            on=['user_id', 'item_id'], 
            how='outer', 
            suffixes=('_als', '_sasrec')
        )
        
        merged['score_als'] = merged['score_als'].fillna(0)
        merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
        merged['final_score'] = (args.w_als * merged['score_als']) + (args.w_sasrec * merged['score_sasrec'])
        print(f"  ✓ {len(merged):,} candidates")
    else:
        print(f"  Warning: SASRec not found, using ALS only")
        merged = als_df.copy()
        merged['rank'] = merged.groupby('user_id').cumcount() + 1
        merged['final_score'] = 1.0 / merged['rank']
    
    # 2. Load user history (simple version)
    print(f"\n[3/4] Loading user history (fast mode)...")
    user_history = load_user_recent_items(args.train_data)
    
    # 3. Apply repeat boost
    print(f"\n[4/4] Applying Repeat Boost...")
    result_df = apply_repeat_boost(merged, user_history, args)
    
    # 4. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Fast optimization completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
