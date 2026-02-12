"""
Phase 1 벡터화 버전: 초고속 Event Type 부스팅
예상: 0.1265 → 0.13~0.135 (+3~5%)
"""

import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

def load_enhanced_history(train_path):
    """이벤트 타입 + 시간 정보 빠르게 로딩"""
    print(f"Loading enhanced history from {train_path}...")
    
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type', 'event_time'])
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    print(f"Total interactions: {len(df):,}")
    
    # 1. 기본 카운트
    print("  Computing interaction counts...")
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    # 2. 이벤트 타입 체크 (purchase/cart 여부)
    print("  Identifying purchase/cart events...")
    has_purchase = df[df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
    has_purchase['has_purchase'] = True
    
    has_cart = df[df['event_type'] == 'cart'][['user_id', 'item_id']].drop_duplicates()
    has_cart['has_cart'] = True
    
    # 3. 최근 시간
    print("  Computing recency...")
    last_time = df.groupby(['user_id', 'item_id'])['event_time'].max().reset_index()
    last_time.columns = ['user_id', 'item_id', 'last_time']
    
    user_last_time = df.groupby('user_id')['event_time'].max().reset_index()
    user_last_time.columns = ['user_id', 'user_last_time']
    
    # 4. 모두 병합
    print("  Merging all features...")
    hist_df = count_df
    hist_df = hist_df.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(user_last_time, on='user_id', how='left')
    
    hist_df['has_purchase'] = hist_df['has_purchase'].fillna(False)
    hist_df['has_cart'] = hist_df['has_cart'].fillna(False)
    
    # 5. 시간 차이 계산 (hours)
    hist_df['hours_since'] = (hist_df['user_last_time'] - hist_df['last_time']).dt.total_seconds() / 3600
    
    print(f"✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df

def apply_phase1_boost(candidates_df, history_df):
    """Phase 1: Event Type + Recency 부스팅 (완전 벡터화)"""
    print("Applying Phase 1 boosts (vectorized)...")
    
    # 1. 기본 Repeat Boost
    print("  Calculating repeat boost...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.0
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.5
    
    # 2. Event Type 차등 (핵심!)
    print("  Applying event type weighting...")
    history_df.loc[history_df['has_purchase'], 'repeat_boost'] *= 1.8
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'repeat_boost'] *= 1.4
    
    # 3. Recency 부스팅
    print("  Applying recency weighting...")
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 2.0
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.5
    history_df.loc[(history_df['hours_since'] > 24) & (history_df['hours_since'] <= 72), 'recency_boost'] = 1.2
    
    # 4. 최종 부스트 = repeat × recency
    history_df['final_boost'] = history_df['repeat_boost'] * history_df['recency_boost']
    
    # 5. Top-3 추가 부스팅 (purchase + 2회 이상 OR 3회 이상)
    print("  Marking top-3 candidates...")
    history_df['is_top3'] = (
        ((history_df['has_purchase']) & (history_df['count'] >= 2)) |
        (history_df['count'] >= 3)
    )
    history_df.loc[history_df['is_top3'], 'final_boost'] *= 1.5
    
    # 6. 후보와 병합
    print("  Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'final_boost']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['final_boost'] = merged['final_boost'].fillna(1.0)
    
    # 7. 최종 점수 계산
    print("  Computing final scores...")
    merged['final_score'] = merged['final_score'] * merged['final_boost']
    
    # 8. Top-10 선택
    print("  Selecting Top-10 per user...")
    result_df = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id', sort=False)
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Vectorized: Event Type + Recency")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase1.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 1 Vectorized: Event Type + Top-3")
    print("Expected: 0.1265 → 0.13~0.135 (+3~5%)")
    print("="*60)
    
    # 1. Load model outputs
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    print(f"  ✓ ALS: {als_df['user_id'].nunique():,} users")
    
    if os.path.exists(args.sasrec_output):
        sasrec_df = pd.read_csv(args.sasrec_output)
        print(f"  ✓ SASRec: {sasrec_df['user_id'].nunique():,} users")
        
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
        merged = als_df.copy()
        merged['rank'] = merged.groupby('user_id').cumcount() + 1
        merged['final_score'] = 1.0 / merged['rank']
    
    # 2. Load enhanced history
    print(f"\n[3/4] Loading enhanced history...")
    history_df = load_enhanced_history(args.train_data)
    
    # 3. Apply Phase 1
    print(f"\n[4/4] Applying Phase 1 optimization...")
    result_df = apply_phase1_boost(merged, history_df)
    
    # 4. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Phase 1 completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print(f"🎯 Expected nDCG@10: 0.13~0.135")
    print("="*60)

if __name__ == "__main__":
    main()
