"""
4-Way Ensemble + Phase 1+ Enhanced Boost
==========================================
4-Way(0.1341) + 3-Way enhanced의 강화 부스트 결합

목표: 모델 다양성 + 부스트 강화의 시너지
예상: 0.1345 - 0.1355
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
    
    print(f"  ✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df

def apply_phase1_enhanced_boost(candidates_df, history_df):
    """Phase 1+ Enhanced: 4-Way에 맞게 강화된 부스트"""
    print("Applying Phase 1+ Enhanced boosts for 4-Way...")
    
    # 1. 기본 Repeat Boost (강화)
    print("  Calculating enhanced repeat boost...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.4  # 1.3 → 1.4
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.1  # 2.0 → 2.1
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.6  # 2.5 → 2.6
    
    # 2. Event Type 차등 (강화)
    print("  Applying enhanced event type weighting...")
    history_df.loc[history_df['has_purchase'], 'repeat_boost'] *= 1.9  # 1.8 → 1.9
    history_df.loc[(~history_df['has_purchase']) & (history_df['has_cart']), 'repeat_boost'] *= 1.5  # 1.4 → 1.5
    
    # 3. Recency 부스팅 (강화)
    print("  Applying enhanced recency weighting...")
    history_df['recency_boost'] = 1.0
    history_df.loc[history_df['hours_since'] <= 1, 'recency_boost'] = 2.1  # 2.0 → 2.1
    history_df.loc[(history_df['hours_since'] > 1) & (history_df['hours_since'] <= 24), 'recency_boost'] = 1.6  # 1.5 → 1.6
    history_df.loc[(history_df['hours_since'] > 24) & (history_df['hours_since'] <= 72), 'recency_boost'] = 1.3  # 1.2 → 1.3
    
    # 4. 최종 부스트 = repeat × recency
    history_df['final_boost'] = history_df['repeat_boost'] * history_df['recency_boost']
    
    # 5. Top-3 추가 부스팅 (강화)
    print("  Marking top-3 candidates...")
    history_df['is_top3'] = (
        ((history_df['has_purchase']) & (history_df['count'] >= 2)) |
        (history_df['count'] >= 3)
    )
    history_df.loc[history_df['is_top3'], 'final_boost'] *= 1.6  # 1.5 → 1.6
    
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
    
    print(f"  ✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def build_4way_ensemble(als_df, sasrec_df, xgb_df, catboost_df, weights, K=60):
    """4-Way 앙상블 빌드"""
    print(f"\nBuilding 4-way ensemble...")
    print(f"  Weights: ALS={weights['als']:.2f}, SASRec={weights['sasrec']:.2f}, XGB={weights['xgb']:.2f}, CatBoost={weights['catboost']:.2f}")
    
    # Rank-based scoring
    for name, df in [('als', als_df), ('sasrec', sasrec_df), ('xgb', xgb_df), ('catboost', catboost_df)]:
        df['rank'] = df.groupby('user_id').cumcount() + 1
        df[f'score_{name}'] = 1.0 / (K + df['rank'])
    
    # Merge
    merged = als_df[['user_id', 'item_id', 'score_als']].merge(
        sasrec_df[['user_id', 'item_id', 'score_sasrec']],
        on=['user_id', 'item_id'], how='outer'
    ).merge(
        xgb_df[['user_id', 'item_id', 'score_xgb']],
        on=['user_id', 'item_id'], how='outer'
    ).merge(
        catboost_df[['user_id', 'item_id', 'score_catboost']],
        on=['user_id', 'item_id'], how='outer'
    )
    
    # Fill NaN
    merged['score_als'] = merged['score_als'].fillna(0)
    merged['score_sasrec'] = merged['score_sasrec'].fillna(0)
    merged['score_xgb'] = merged['score_xgb'].fillna(0)
    merged['score_catboost'] = merged['score_catboost'].fillna(0)
    
    # Weighted score
    merged['final_score'] = (
        weights['als'] * merged['score_als'] +
        weights['sasrec'] * merged['score_sasrec'] +
        weights['xgb'] * merged['score_xgb'] +
        weights['catboost'] * merged['score_catboost']
    )
    
    print(f"  ✓ Total candidates: {len(merged):,}")
    print(f"  ✓ Avg candidates/user: {len(merged) / merged['user_id'].nunique():.1f}")
    
    return merged[['user_id', 'item_id', 'final_score']]

def main():
    parser = argparse.ArgumentParser(description="4-Way + Phase 1+ Enhanced Boost")
    
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--xgb_output", default="../output/output_reranked_21.csv", type=str)
    parser.add_argument("--catboost_output", default="../output/output_catboost_final_29.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_quad_enhanced.csv", type=str)
    
    parser.add_argument("--w_als", default=0.5, type=float)
    parser.add_argument("--w_sasrec", default=0.25, type=float)
    parser.add_argument("--w_xgb", default=0.10, type=float)
    parser.add_argument("--w_catboost", default=0.15, type=float)
    parser.add_argument("--K", default=60, type=int)
    
    args = parser.parse_args()
    
    print("="*60)
    print("4-Way + Phase 1+ Enhanced Boost")
    print("Target: 0.1341 → 0.1345~0.1355")
    print("="*60)
    
    # 1. Load
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgb_output)
    catboost_df = pd.read_csv(args.catboost_output)
    
    # 2. Ensemble
    weights = {
        'als': args.w_als,
        'sasrec': args.w_sasrec,
        'xgb': args.w_xgb,
        'catboost': args.w_catboost
    }
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    
    print(f"\n[2/4] Building ensemble...")
    candidates_df = build_4way_ensemble(als_df, sasrec_df, xgb_df, catboost_df, weights, args.K)
    
    # 3. Load history
    print(f"\n[3/4] Loading history...")
    history_df = load_enhanced_history(args.train_data)
    
    # 4. Apply enhanced boost
    print(f"\n[4/4] Applying Phase 1+ enhanced boost...")
    result_df = apply_phase1_enhanced_boost(candidates_df, history_df)
    
    # 5. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    n_users = result_df['user_id'].nunique()
    n_rows = len(result_df)
    
    print("\n" + "="*60)
    print(f"✅ 4-Way + Phase 1+ Enhanced completed!")
    print(f"📊 Total predictions: {n_rows:,}")
    print(f"👥 Users: {n_users:,}")
    print(f"💾 Saved to: {args.output_path}")
    
    if n_users == 638257 and n_rows == 6382570:
        print(f"✅ Validation PASSED")
    
    print("="*60)

if __name__ == "__main__":
    main()
