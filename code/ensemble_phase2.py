"""
Phase 2: Category Consistency + Diversity 최적화
현재: 0.1330 → 목표: 0.14~0.145 (+5~10%)
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
from tqdm import tqdm

def load_full_features(train_path):
    """모든 피처 로딩: Event + Time + Category"""
    print(f"Loading all features from {train_path}...")
    
    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type', 'event_time', 'category_code'])
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    print(f"Total interactions: {len(df):,}")
    
    # 1. 기본 카운트 + 이벤트 타입
    print("  Computing base features...")
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    has_purchase = df[df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
    has_purchase['has_purchase'] = True
    
    has_cart = df[df['event_type'] == 'cart'][['user_id', 'item_id']].drop_duplicates()
    has_cart['has_cart'] = True
    
    # 2. 시간 정보
    print("  Computing temporal features...")
    last_time = df.groupby(['user_id', 'item_id'])['event_time'].max().reset_index()
    last_time.columns = ['user_id', 'item_id', 'last_time']
    
    user_last_time = df.groupby('user_id')['event_time'].max().reset_index()
    user_last_time.columns = ['user_id', 'user_last_time']
    
    # 3. 카테고리 정보
    print("  Computing category features...")
    
    # 아이템 → 카테고리 매핑
    item_cat = df[df['category_code'].notna()][['item_id', 'category_code']].drop_duplicates()
    
    # 유저별 최근 카테고리 (최근 5개)
    df_sorted = df[df['category_code'].notna()].sort_values(['user_id', 'event_time'])
    recent_cats = df_sorted.groupby('user_id')['category_code'].apply(lambda x: x.tail(5).tolist()).reset_index()
    recent_cats.columns = ['user_id', 'recent_categories']
    
    # 4. 모두 병합
    print("  Merging all features...")
    hist_df = count_df
    hist_df = hist_df.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(user_last_time, on='user_id', how='left')
    hist_df = hist_df.merge(item_cat, on='item_id', how='left')
    
    hist_df['has_purchase'] = hist_df['has_purchase'].fillna(False)
    hist_df['has_cart'] = hist_df['has_cart'].fillna(False)
    
    hist_df['hours_since'] = (hist_df['user_last_time'] - hist_df['last_time']).dt.total_seconds() / 3600
    
    print(f"✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df, recent_cats, item_cat

def apply_phase2_boost(candidates_df, history_df, recent_cats_df, item_cat_df):
    """Phase 2: Phase1 + Category Consistency + Diversity"""
    print("Applying Phase 2 boosts (vectorized)...")
    
    # === Phase 1 부스팅 (재사용) ===
    print("  [Phase 1] Calculating repeat + event type + recency...")
    
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
    
    # 4. Top-3
    history_df['is_top3'] = (
        ((history_df['has_purchase']) & (history_df['count'] >= 2)) |
        (history_df['count'] >= 3)
    )
    
    # === Phase 2 추가: Category Consistency ===
    print("  [Phase 2] Applying category consistency...")
    
    # 카테고리 부스트 계산
    def calc_category_boost(row, recent_cats_dict):
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
    
    # 딕셔너리로 변환
    recent_cats_dict = dict(zip(recent_cats_df['user_id'], recent_cats_df['recent_categories']))
    
    tqdm.pandas(desc="  Category boost")
    history_df['category_boost'] = history_df.progress_apply(lambda row: calc_category_boost(row, recent_cats_dict), axis=1)
    
    # 5. 최종 부스트 = repeat × recency × category
    history_df['final_boost'] = history_df['repeat_boost'] * history_df['recency_boost'] * history_df['category_boost']
    
    # Top-3 추가 부스트
    history_df.loc[history_df['is_top3'], 'final_boost'] *= 1.5
    
    # === 후보와 병합 ===
    print("  Merging with candidates...")
    merged = candidates_df.merge(
        history_df[['user_id', 'item_id', 'final_boost', 'category_code']],
        on=['user_id', 'item_id'],
        how='left'
    )
    merged['final_boost'] = merged['final_boost'].fillna(1.0)
    
    # 최종 점수
    merged['final_score'] = merged['final_score'] * merged['final_boost']
    
    # === Phase 2 추가: Diversity Penalty (벡터화) ===
    print("  [Phase 2] Applying diversity penalty (vectorized)...")
    
    # 카테고리별 rank를 계산하여 같은 카테고리가 반복되면 약간의 페널티 적용
    merged['category_rank'] = merged.groupby(['user_id', 'category_code'])['final_score'].rank(method='first', ascending=False)
    
    # 카테고리가 3번 이상 나타나면 점수를 약간 감소 (4-10위에서만)
    merged['rank'] = merged.groupby('user_id')['final_score'].rank(method='first', ascending=False)
    
    # 4-10위에서 같은 카테고리의 3번째 이후 아이템에 페널티
    diversity_penalty = 1.0
    diversity_penalty = np.where(
        (merged['rank'] > 3) & (merged['category_rank'] > 3),
        0.95,  # 5% 페널티
        diversity_penalty
    )
    
    merged['final_score'] = merged['final_score'] * diversity_penalty
    
    # === 유저별 Top-10 선택 (벡터화) ===
    print("  Selecting Top-10 per user (vectorized)...")
    
    # 벡터화된 Top-10 선택
    result_df = (merged
                 .sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id')
                 .head(10)
                if row['item_id'] not in [p['item_id'] for p in diverse_picks]:
                    diverse_picks.append(row)
                    if len(diverse_picks) >= 7:
                        break
        
        final_10 = pd.concat([top3, pd.DataFrame(diverse_picks).head(7)], ignore_index=True)
        results.append(final_10[['user_id', 'item_id']])
    
    result_df = pd.concat(results, ignore_index=True)
    print(f"✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Category + Diversity")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_phase2.csv", type=str)
    parser.add_argument("--w_als", default=0.7, type=float)
    parser.add_argument("--w_sasrec", default=0.3, type=float)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Phase 2: Category Consistency + Diversity")
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
        merged['final_score'] = (args.w_als * merged['score_als']) + (args.w_sasrec * merged['score_sasrec'])
        print(f"  ✓ {len(merged):,} candidates")
    else:
        merged = als_df.copy()
        merged['rank'] = merged.groupby('user_id').cumcount() + 1
        merged['final_score'] = 1.0 / merged['rank']
    
    # 2. Load full features
    print(f"\n[3/4] Loading all features (Event + Time + Category)...")
    history_df, recent_cats_df, item_cat_df = load_full_features(args.train_data)
    
    # 3. Apply Phase 2
    print(f"\n[4/4] Applying Phase 2 optimization...")
    result_df = apply_phase2_boost(merged, history_df, recent_cats_df, item_cat_df)
    
    # 4. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Phase 2 completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print(f"🎯 Expected nDCG@10: 0.14~0.145")
    print("="*60)

if __name__ == "__main__":
    main()
