"""
Phase 3: Top-3 집중 최적화 + Position-aware Boosting
Phase 2 실패 분석: Category는 효과 없음, Top-3에 집중해야 함
예상: 0.1331 → 0.14+ (+5% 목표)
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

def apply_phase3_boost(candidates_df, history_df):
    """Phase 3: Top-3 집중 + Position-aware (완전 벡터화)"""
    print("Applying Phase 3 boosts (Top-3 focused, vectorized)...")
    
    # === 1. 기본 Feature Boost (Phase 1 강화) ===
    print("  [1/4] Calculating base boost (Repeat + Event + Recency)...")
    
    # 1-1. Repeat Boost (더 강화)
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.2  # 2.0 → 2.2
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.8  # 2.5 → 2.8
    
    # 1-2. Event Type Boost (더 강화)
    history_df['event_boost'] = 1.0
    history_df.loc[history_df['has_purchase'], 'event_boost'] = 2.2  # 1.8 → 2.2
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'event_boost'] = 1.6  # 1.4 → 1.6
    
    # 1-3. Recency Boost (유지)
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 2.0
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.5
    
    # 1-4. **NEW: Purchase Priority (구매는 무조건 상위)**
    history_df['purchase_priority'] = 0.0
    history_df.loc[history_df['has_purchase'], 'purchase_priority'] = 100.0  # 구매는 +100점
    
    # 1-5. 최종 베이스 부스트
    history_df['base_boost'] = (history_df['repeat_boost'] * 
                                 history_df['event_boost'] * 
                                 history_df['recency_boost'])
    
    # === 2. 후보와 병합 ===
    print("  [2/4] Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'base_boost', 'purchase_priority']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['base_boost'] = merged['base_boost'].fillna(1.0)
    merged['purchase_priority'] = merged['purchase_priority'].fillna(0.0)
    
    # 베이스 점수 적용
    merged['boosted_score'] = merged['final_score'] * merged['base_boost'] + merged['purchase_priority']
    
    # === 3. **NEW Phase 3: Position-aware Top-3 Boosting** ===
    print("  [3/4] Applying position-aware Top-3 boost (vectorized)...")
    
    # 각 유저별 순위 계산 (boosted_score 기준)
    merged['temp_rank'] = merged.groupby('user_id')['boosted_score'].rank(method='first', ascending=False)
    
    # Top-3 추가 부스트 (매우 강력하게)
    position_boost = np.where(merged['temp_rank'] == 1, 2.0,      # 1위: 2.0배
                     np.where(merged['temp_rank'] == 2, 1.8,      # 2위: 1.8배
                     np.where(merged['temp_rank'] == 3, 1.6,      # 3위: 1.6배
                     np.where(merged['temp_rank'] <= 10, 1.2,     # 4-10위: 1.2배
                              1.0))))                              # 나머지: 1.0배
    
    merged['final_score'] = merged['boosted_score'] * position_boost
    
    # === 4. Top-10 선택 ===
    print("  [4/4] Selecting Top-10 per user (vectorized)...")
    result_df = (merged
                 .sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id')
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Top-3 Focused")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase3.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 3: Top-3 Focused + Position-aware Boosting")
    print("Current: 0.1331 → Target: 0.14+ (+5%)")
    print("Strategy: Stronger boosts + Position weighting")
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
    
    # 4. Apply Phase 3 boost
    print(f"\n[4/4] Applying Phase 3 optimization...")
    result_df = apply_phase3_boost(candidates_df, history_df)
    
    # 5. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ Phase 3 optimization completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print()
    print("Key improvements:")
    print("  • Stronger repeat boost: 2.8x (was 2.5x)")
    print("  • Stronger event boost: 2.2x for purchase (was 1.8x)")
    print("  • Purchase priority: +100 bonus points")
    print("  • Position-aware: Top-3 get 1.6~2.0x extra boost")
    print("="*60)

if __name__ == "__main__":
    main()
