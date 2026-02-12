"""
Phase 2 벡터화 버전: 초고속 Category + Diversity
예상: 0.1330 → 0.14~0.145 (+5~10%)
"""

import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

def load_full_history(train_path):
    """이벤트 타입 + 시간 + 카테고리 정보 빠르게 로딩"""
    print(f"Loading full history from {train_path}...")
    
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type', 'event_time', 'category_code'])
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
    
    # 4. 카테고리 정보 (아이템별 가장 최근 카테고리)
    print("  Loading category info...")
    item_cat = df.sort_values('event_time').groupby('item_id')['category_code'].last().reset_index()
    item_cat.columns = ['item_id', 'category_code']
    
    # 5. 유저별 최근 카테고리 (최근 5개)
    print("  Computing recent categories per user...")
    recent_df = (df[df['category_code'].notna()]
                 .sort_values(['user_id', 'event_time'], ascending=[True, False])
                 .groupby('user_id')['category_code']
                 .apply(lambda x: list(x.unique()[:5]))
                 .reset_index())
    recent_df.columns = ['user_id', 'recent_categories']
    
    # 6. 모두 병합
    print("  Merging all features...")
    hist_df = count_df
    hist_df = hist_df.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(user_last_time, on='user_id', how='left')
    hist_df = hist_df.merge(item_cat, on='item_id', how='left')
    
    hist_df['has_purchase'] = hist_df['has_purchase'].fillna(False)
    hist_df['has_cart'] = hist_df['has_cart'].fillna(False)
    
    # 7. 시간 차이 계산 (hours)
    hist_df['hours_since'] = (hist_df['user_last_time'] - hist_df['last_time']).dt.total_seconds() / 3600
    
    print(f"✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df, recent_df

def apply_phase2_boost(candidates_df, history_df, recent_cats_df):
    """Phase 2: Event Type + Recency + Category 부스팅 (완전 벡터화)"""
    print("Applying Phase 2 boosts (vectorized)...")
    
    # 1. 기본 Repeat Boost
    print("  Calculating repeat boost...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.0
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.5
    
    # 2. Event Type 차등 (핵심!)
    print("  Calculating event type boost...")
    history_df['event_boost'] = 1.0
    history_df.loc[history_df['has_purchase'], 'event_boost'] = 1.8
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'event_boost'] = 1.4
    
    # 3. Recency Boost (1시간 이내)
    print("  Calculating recency boost...")
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 2.0
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.5
    
    # 4. **NEW Phase 2: Category Consistency Boost**
    print("  Calculating category boost (vectorized)...")
    
    # 최근 카테고리를 dict로 변환
    recent_cats_dict = dict(zip(recent_cats_df['user_id'], recent_cats_df['recent_categories']))
    
    def calc_category_boost(row):
        user_id = row['user_id']
        category = row['category_code']
        
        if pd.isna(category) or user_id not in recent_cats_dict:
            return 1.0
        
        recent_cats = recent_cats_dict[user_id]
        
        if category in recent_cats[:2]:  # 최근 2개 카테고리
            return 1.6
        elif category in recent_cats[:4]:  # 최근 4개 카테고리
            return 1.3
        elif category in recent_cats:  # 최근 5개 카테고리
            return 1.1
        
        return 1.0
    
    # tqdm으로 진행률 표시
    tqdm.pandas(desc="  Category boost")
    history_df['category_boost'] = history_df.progress_apply(calc_category_boost, axis=1)
    
    # 5. 최종 부스트 = repeat × event × recency × category
    print("  Computing final boost...")
    history_df['final_boost'] = (history_df['repeat_boost'] * 
                                  history_df['event_boost'] * 
                                  history_df['recency_boost'] * 
                                  history_df['category_boost'])
    
    # 6. 후보와 병합
    print("  Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'final_boost', 'category_code']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['final_boost'] = merged['final_boost'].fillna(1.0)
    
    # 7. 최종 점수
    merged['final_score'] = merged['final_score'] * merged['final_boost']
    
    # 8. **Phase 2 NEW: Diversity Penalty (벡터화)**
    print("  Applying diversity penalty (vectorized)...")
    
    # 카테고리별 rank 계산
    merged['category_rank'] = merged.groupby(['user_id', 'category_code'])['final_score'].rank(method='first', ascending=False)
    
    # 전체 rank 계산
    merged['item_rank'] = merged.groupby('user_id')['final_score'].rank(method='first', ascending=False)
    
    # 4-10위에서 같은 카테고리의 4번째 이후 아이템에 페널티
    diversity_penalty = np.where(
        (merged['item_rank'] > 3) & (merged['category_rank'] >= 4),
        0.90,  # 10% 페널티
        1.0
    )
    
    merged['final_score'] = merged['final_score'] * diversity_penalty
    
    # 9. Top-10 선택 (벡터화)
    print("  Selecting Top-10 per user (vectorized)...")
    result_df = (merged
                 .sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id')
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Fast: Category + Diversity")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase2.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 2 Fast: Category Consistency + Diversity")
    print("Current: 0.1330 → Target: 0.14~0.145 (+5~10%)")
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
    
    # 3. Load user history with all features
    print(f"\n[3/4] Loading full history (Event + Time + Category)...")
    history_df, recent_cats_df = load_full_history(args.train_data)
    
    # 4. Apply Phase 2 boost
    print(f"\n[4/4] Applying Phase 2 optimization...")
    result_df = apply_phase2_boost(candidates_df, history_df, recent_cats_df)
    
    # 5. Save
    result_df.to_csv(args.output_path, index=False)
    
    print()
    print("="*60)
    print("✅ Phase 2 Fast optimization completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
