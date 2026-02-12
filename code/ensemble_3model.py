"""
3모델 앙상블 + Phase 1 Post-processing
ALS + SASRec + EASE → Phase 1 boost 적용
핵심: 후보 풀 품질 향상으로 0.1330 돌파
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
    
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    has_purchase = df[df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
    has_purchase['has_purchase'] = True
    
    has_cart = df[df['event_type'] == 'cart'][['user_id', 'item_id']].drop_duplicates()
    has_cart['has_cart'] = True
    
    last_time = df.groupby(['user_id', 'item_id'])['event_time'].max().reset_index()
    last_time.columns = ['user_id', 'item_id', 'last_time']
    
    user_last_time = df.groupby('user_id')['event_time'].max().reset_index()
    user_last_time.columns = ['user_id', 'user_last_time']
    
    hist_df = count_df
    hist_df = hist_df.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(user_last_time, on='user_id', how='left')
    
    hist_df['has_purchase'] = hist_df['has_purchase'].fillna(False)
    hist_df['has_cart'] = hist_df['has_cart'].fillna(False)
    
    hist_df['hours_since'] = (hist_df['user_last_time'] - hist_df['last_time']).dt.total_seconds() / 3600
    
    print(f"✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df

def apply_phase1_boost(candidates_df, history_df):
    """Phase 1 부스팅 (검증된 최적 값 - 0.1330 달성)"""
    print("Applying Phase 1 boosts (proven optimal)...")
    
    # 1. Repeat Boost
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.0
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.5
    
    # 2. Event Type
    history_df.loc[history_df['has_purchase'], 'repeat_boost'] *= 1.8
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'repeat_boost'] *= 1.4
    
    # 3. Recency
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 2.0
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.5
    history_df.loc[(history_df['hours_since'] > 24) & (history_df['hours_since'] <= 72), 'recency_boost'] = 1.2
    
    # 4. 최종 부스트
    history_df['final_boost'] = history_df['repeat_boost'] * history_df['recency_boost']
    
    # 5. Top-3 추가 부스팅
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
    
    # 7. 최종 점수
    merged['final_score'] = merged['final_score'] * merged['final_boost']
    
    # 8. Top-10 선택
    print("  Selecting Top-10 per user...")
    result_df = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id', sort=False)
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def create_3model_ensemble(als_path, sasrec_path, ease_path, w_als, w_sasrec, w_ease):
    """ALS + SASRec + EASE 3모델 앙상블"""
    print(f"\nCreating 3-model ensemble...")
    print(f"  Weights: ALS={w_als}, SASRec={w_sasrec}, EASE={w_ease}")
    
    # Load all models
    als_df = pd.read_csv(als_path)
    print(f"  ✓ ALS: {als_df['user_id'].nunique():,} users")
    
    sasrec_df = pd.read_csv(sasrec_path)
    print(f"  ✓ SASRec: {sasrec_df['user_id'].nunique():,} users")
    
    ease_df = pd.read_csv(ease_path)
    print(f"  ✓ EASE: {ease_df['user_id'].nunique():,} users")
    
    # Rank-based scoring (1/rank)
    als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
    als_df['score'] = 1.0 / als_df['rank']
    
    sasrec_df['rank'] = sasrec_df.groupby('user_id').cumcount() + 1
    sasrec_df['score'] = 1.0 / sasrec_df['rank']
    
    ease_df['rank'] = ease_df.groupby('user_id').cumcount() + 1
    ease_df['score'] = 1.0 / ease_df['rank']
    
    # Merge all three
    print("  Merging ALS + SASRec...")
    merged = pd.merge(
        als_df[['user_id', 'item_id', 'score']],
        sasrec_df[['user_id', 'item_id', 'score']],
        on=['user_id', 'item_id'],
        how='outer',
        suffixes=('_als', '_sasrec')
    )
    
    print("  Merging with EASE...")
    merged = pd.merge(
        merged,
        ease_df[['user_id', 'item_id', 'score']].rename(columns={'score': 'score_ease'}),
        on=['user_id', 'item_id'],
        how='outer'
    )
    
    # Fill NaN
    merged['score_als'] = merged['score_als'].fillna(0)
    merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
    merged['score_ease'] = merged['score_ease'].fillna(0)
    
    # Weighted combination
    merged['final_score'] = (w_als * merged['score_als'] + 
                              w_sasrec * merged['score_sasrec'] + 
                              w_ease * merged['score_ease'])
    
    candidates_df = merged[['user_id', 'item_id', 'final_score']]
    
    print(f"  ✓ {len(candidates_df):,} candidates from 3 models")
    print(f"  ✓ Unique items: {candidates_df['item_id'].nunique():,}")
    
    return candidates_df

def main():
    parser = argparse.ArgumentParser(description="3-Model Ensemble + Phase 1")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--ease_output", default="../output/output_ease_24.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_3model.csv", type=str)
    parser.add_argument("--w_als", default=0.5, type=float, help="ALS weight")
    parser.add_argument("--w_sasrec", default=0.2, type=float, help="SASRec weight")
    parser.add_argument("--w_ease", default=0.3, type=float, help="EASE weight")
    
    args = parser.parse_args()
    
    print("="*60)
    print("3-Model Ensemble + Phase 1 Post-processing")
    print(f"Weights: ALS={args.w_als}, SASRec={args.w_sasrec}, EASE={args.w_ease}")
    print("Base: Phase 1 boost (proven 0.1330)")
    print("Target: 0.135~0.145 (new candidates from EASE)")
    print("="*60)
    
    # 1. Create 3-model ensemble
    candidates_df = create_3model_ensemble(
        args.als_output, args.sasrec_output, args.ease_output,
        args.w_als, args.w_sasrec, args.w_ease
    )
    
    # 2. Load user history
    print(f"\nLoading enhanced history (Event + Time)...")
    history_df = load_enhanced_history(args.train_data)
    
    # 3. Apply Phase 1 boost
    print(f"\nApplying Phase 1 optimization...")
    result_df = apply_phase1_boost(candidates_df, history_df)
    
    # 4. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ 3-Model Ensemble completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"⚖️  ALS={args.w_als}, SASRec={args.w_sasrec}, EASE={args.w_ease}")
    print(f"💾 Saved to: {args.output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
