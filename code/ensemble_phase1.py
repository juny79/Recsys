"""
Phase 1: Event Type 차등 Boost + Top-3 집중 최적화
예상: 0.1265 → 0.13~0.135 (+3~5%)
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

def load_user_history_with_events(train_path):
    """이벤트 타입 정보 포함 히스토리 로딩"""
    print(f"Loading history with event types from {train_path}...")
    
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type', 'event_time'])
    print(f"Total interactions: {len(df):,}")
    
    # 이벤트 시간 파싱
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 유저별 처리
    user_history = {}
    
    print("Building enhanced history...")
    for user_id, user_df in tqdm(df.groupby('user_id'), desc="Processing"):
        item_data = {}
        
        for item_id, item_df in user_df.groupby('item_id'):
            item_data[item_id] = {
                'count': len(item_df),
                'event_types': item_df['event_type'].tolist(),
                'last_time': item_df['event_time'].max(),
                'has_purchase': 'purchase' in item_df['event_type'].values,
                'has_cart': 'cart' in item_df['event_type'].values
            }
        
        user_history[user_id] = {
            'items': item_data,
            'last_interaction': user_df['event_time'].max()
        }
    
    print(f"✓ Loaded history for {len(user_history):,} users")
    return user_history

def calculate_enhanced_boost(item_id, user_hist):
    """Phase 1: Event Type 차등 + Recency 고려"""
    if item_id not in user_hist['items']:
        return 1.0
    
    item_data = user_hist['items'][item_id]
    count = item_data['count']
    has_purchase = item_data['has_purchase']
    has_cart = item_data['has_cart']
    last_time = item_data['last_time']
    user_last_time = user_hist['last_interaction']
    
    # 1. 기본 Repeat Boost
    if count >= 3:
        boost = 2.5
    elif count >= 2:
        boost = 2.0
    else:
        boost = 1.3
    
    # 2. Event Type 차등 (핵심!)
    if has_purchase:
        boost *= 1.8  # 구매 이력 → 강력한 시그널
    elif has_cart:
        boost *= 1.4  # 장바구니 이력
    
    # 3. Recency Boost
    hours_since = (user_last_time - last_time).total_seconds() / 3600
    if hours_since <= 1:
        recency = 2.0
    elif hours_since <= 24:
        recency = 1.5
    elif hours_since <= 72:
        recency = 1.2
    else:
        recency = 1.0
    
    boost *= recency
    
    return boost

def apply_phase1_optimization(candidates_df, user_history, args):
    """Phase 1: Event Type + Top-3 최적화"""
    print("Applying Phase 1 optimization...")
    
    # 1. 히스토리를 빠른 조회용 dict로 변환
    print("  Preparing boost data...")
    boost_data = {}
    
    for user_id in tqdm(candidates_df['user_id'].unique(), desc="  Calculating boosts"):
        if user_id not in user_history:
            continue
        
        user_hist = user_history[user_id]
        user_boosts = {}
        
        for item_id in user_hist['items'].keys():
            user_boosts[item_id] = calculate_enhanced_boost(item_id, user_hist)
        
        boost_data[user_id] = user_boosts
    
    # 2. 부스트 적용 (벡터화)
    print("  Applying boosts...")
    
    def get_boost(row):
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id in boost_data and item_id in boost_data[user_id]:
            return boost_data[user_id][item_id]
        return 1.0
    
    tqdm.pandas(desc="  Boosting")
    candidates_df['boost'] = candidates_df.progress_apply(get_boost, axis=1)
    candidates_df['final_score'] = candidates_df['final_score'] * candidates_df['boost']
    
    # 3. Top-3 집중 최적화
    print("  Optimizing Top-3 positions...")
    
    # Top-3 후보 식별 (purchase + 재방문)
    def is_top3_candidate(row):
        user_id = row['user_id']
        item_id = row['item_id']
        
        if user_id in user_history and item_id in user_history[user_id]['items']:
            item_data = user_history[user_id]['items'][item_id]
            # 구매 이력 + 2회 이상 OR 3회 이상 재방문
            if (item_data['has_purchase'] and item_data['count'] >= 2) or item_data['count'] >= 3:
                return True
        return False
    
    tqdm.pandas(desc="  Top-3 marking")
    candidates_df['is_top3'] = candidates_df.progress_apply(is_top3_candidate, axis=1)
    candidates_df.loc[candidates_df['is_top3'], 'final_score'] *= 1.5  # Top-3 추가 부스트
    
    # 4. 유저별 Top-10 선택
    print("  Selecting Top-10 per user...")
    result_df = (candidates_df.sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id', sort=False)
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Event Type + Top-3 Optimization")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase1.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 1: Event Type + Top-3 Optimization")
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
    print(f"\n[3/4] Loading user history with event types...")
    user_history = load_user_history_with_events(args.train_data)
    
    # 3. Apply Phase 1 optimization
    print(f"\n[4/4] Applying Phase 1 optimization...")
    result_df = apply_phase1_optimization(merged, user_history, args)
    
    # 4. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Phase 1 optimization completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print(f"🎯 Expected nDCG@10: 0.13~0.135")
    print("="*60)

if __name__ == "__main__":
    main()
