"""
Phase 5: Recency 세밀화 + Boost 미세 조정
Phase 1 (0.1330) 베이스 유지, 가중치 0.7/0.3 고정
전략: Recency window를 더 세분화하고 boost 값 미세 조정
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

def apply_phase5_boost(candidates_df, history_df):
    """Phase 5: 세밀한 Recency + 미세 조정된 Boost"""
    print("Applying Phase 5 boosts (fine-tuned)...")
    
    # === 1. Repeat Boost (미세 조정) ===
    print("  [1/4] Calculating repeat boost (fine-tuned)...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.1  # 2.0 → 2.1
    history_df.loc[history_df['count'] == 3, 'repeat_boost'] = 2.4  # 2.5 → 2.4
    history_df.loc[history_df['count'] >= 4, 'repeat_boost'] = 2.6  # 2.5 → 2.6 (4회 이상 차별화)
    
    # === 2. Event Type Boost (미세 조정) ===
    print("  [2/4] Calculating event type boost (fine-tuned)...")
    history_df['event_boost'] = 1.0
    history_df.loc[history_df['has_purchase'], 'event_boost'] = 1.9  # 1.8 → 1.9
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'event_boost'] = 1.45  # 1.4 → 1.45
    
    # === 3. Recency Boost (세밀화) ===
    print("  [3/4] Calculating recency boost (fine-grained)...")
    history_df['recency_boost'] = 1.0
    
    # 매우 세밀한 시간 구간
    history_df.loc[history_df['hours_since'] <= 0.5, 'recency_boost'] = 2.2   # 30분 이내 (NEW)
    history_df.loc[(history_df['hours_since'] > 0.5) & (history_df['hours_since'] <= 1), 'recency_boost'] = 2.0   # 1시간
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 3), 'recency_boost'] = 1.7    # 3시간 (NEW)
    history_df.loc[(history_df['hours_since'] > 3) & (history_df['hours_since'] <= 6), 'recency_boost'] = 1.5    # 6시간 (NEW)
    history_df.loc[(history_df['hours_since'] > 6) & (history_df['hours_since'] <= 12), 'recency_boost'] = 1.35  # 12시간 (NEW)
    history_df.loc[(history_df['hours_since'] > 12) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.2  # 24시간
    history_df.loc[(history_df['hours_since'] > 24) & (history_df['hours_since'] <= 72), 'recency_boost'] = 1.1  # 3일 (NEW)
    
    # === 4. 최종 부스트 ===
    print("  [4/4] Computing final boost...")
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
    parser = argparse.ArgumentParser(description="Phase 5: Fine-tuned Recency")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase5.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 5: Fine-tuned Recency + Boost Micro-adjustment")
    print("Base: Phase 1 (0.1330) with optimal 0.7/0.3 weights")
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
    
    # 4. Apply Phase 5 boost
    print(f"\n[4/4] Applying Phase 5 optimization...")
    result_df = apply_phase5_boost(candidates_df, history_df)
    
    # 5. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ Phase 5 optimization completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print()
    print("Key improvements:")
    print("  • Repeat boost: 2.1/2.4/2.6 (count 2/3/4+)")
    print("  • Event boost: 1.9 (purchase), 1.45 (cart)")
    print("  • Recency: 7 time windows (30m/1h/3h/6h/12h/24h/3d)")
    print("  • Ensemble: 0.7/0.3 (proven optimal)")
    print("="*60)

if __name__ == "__main__":
    main()
