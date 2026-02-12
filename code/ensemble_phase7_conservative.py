"""
Phase 7: Conservative Boost (Phase 1보다 약하게)
가설: Phase 1이 오히려 over-fitting일 수 있음
전략: 더 약한 boost로 generalization 향상
예상: 0.1330 → 0.135~0.14 (+1.5~5%)
"""

import pandas as pd
import numpy as np
import argparse
import os

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

def apply_conservative_boost(candidates_df, history_df):
    """Phase 1보다 약한 boost (더 보수적)"""
    print("Applying conservative boosts (weaker than Phase 1)...")
    
    # 1. Repeat Boost (Phase 1보다 약하게)
    print("  Calculating repeat boost (conservative)...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.2   # 1.3 → 1.2
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 1.7   # 2.0 → 1.7
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.1   # 2.5 → 2.1
    
    # 2. Event Type Boost (Phase 1보다 약하게)
    print("  Calculating event type boost (conservative)...")
    history_df['event_boost'] = 1.0
    history_df.loc[history_df['has_purchase'], 'event_boost'] = 1.5   # 1.8 → 1.5
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'event_boost'] = 1.25  # 1.4 → 1.25
    
    # 3. Recency Boost (Phase 1보다 약하게)
    print("  Calculating recency boost (conservative)...")
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 1.6   # 2.0 → 1.6
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.3  # 1.5 → 1.3
    
    # 4. 최종 부스트
    print("  Computing final boost...")
    history_df['final_boost'] = (history_df['repeat_boost'] * 
                                  history_df['event_boost'] * 
                                  history_df['recency_boost'])
    
    # 5. 후보와 병합
    print("  Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'final_boost']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['final_boost'] = merged['final_boost'].fillna(1.0)
    
    # 6. 최종 점수
    merged['final_score'] = merged['final_score'] * merged['final_boost']
    
    # 7. Top-10 선택
    print("  Selecting Top-10 per user...")
    result_df = (merged
                 .sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id')
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Conservative")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase7_conservative.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 7: Conservative Boost Strategy")
    print("Hypothesis: Phase 1 over-fits, weaker boost generalizes better")
    print("Base: 0.7/0.3 ensemble, weaker boost than Phase 1")
    print("Target: 0.135~0.14 (+1.5~5%)")
    print("="*60)
    
    # 1. Load model outputs
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    print(f"  ✓ ALS: {als_df['user_id'].nunique():,} users")
    
    if os.path.exists(args.sasrec_output):
        sasrec_df = pd.read_csv(args.sasrec_output)
        print(f"  ✓ SASRec: {sasrec_df['user_id'].nunique():,} users")
        
        print(f"\n[2/4] Creating base ensemble (0.7/0.3)...")
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
        
        merged['final_score'] = args.w_als * merged['score_als'] + args.w_sasrec * merged['score_sasrec']
        
        candidates_df = merged[['user_id', 'item_id', 'final_score']]
    else:
        print(f"  ⚠ SASRec output not found, using ALS only...")
        print(f"\n[2/4] Using ALS predictions...")
        als_df['rank'] = als_df.groupby('user_id').cumcount() + 1
        als_df['final_score'] = 1.0 / als_df['rank']
        candidates_df = als_df[['user_id', 'item_id', 'final_score']]
    
    print(f"  ✓ {len(candidates_df):,} candidates")
    
    # 3. Load user history
    print(f"\n[3/4] Loading enhanced history (Event + Time)...")
    history_df = load_enhanced_history(args.train_data)
    
    # 4. Apply conservative boost
    print(f"\n[4/4] Applying Phase 7 optimization...")
    result_df = apply_conservative_boost(candidates_df, history_df)
    
    # 5. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ Phase 7 (Conservative) completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print()
    print("Conservative boost values:")
    print("  • Repeat: 1.2/1.7/2.1 (vs Phase 1: 1.3/2.0/2.5)")
    print("  • Event: 1.5/1.25 (vs Phase 1: 1.8/1.4)")
    print("  • Recency: 1.6/1.3 (vs Phase 1: 2.0/1.5)")
    print("  • Max combined: ~5.3x (vs Phase 1: ~9x)")
    print("="*60)

if __name__ == "__main__":
    main()
