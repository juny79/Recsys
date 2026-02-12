"""
3-Way / 4-Way Ensemble + Phase 1 Post-Processing
==================================================
핵심 아이디어: 
  - 3-Way Ensemble(ALS+SASRec+XGB)은 0.1196 달성 (Phase 1 부스트 미적용)
  - 2-Way Ensemble + Phase 1 = 0.1330
  - 3-Way는 base가 0.1196으로 더 높으므로, Phase 1 부스트 적용 시 0.1330 돌파 가능

후보 풀 확장:
  - 각 모델 Top-10 outer merge → 최대 30~40개 후보/유저
  - Phase 1 부스트로 재랭킹 → Top-10 선정
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
    
    print(f"  ✓ Processed {len(hist_df):,} user-item pairs")
    
    return hist_df

def apply_phase1_boost(candidates_df, history_df):
    """Phase 1: Event Type + Recency 부스팅 (검증된 0.1330 로직 그대로)"""
    print("Applying Phase 1 boosts (vectorized)...")
    
    # 1. 기본 Repeat Boost
    print("  Calculating repeat boost...")
    history_df['repeat_boost'] = 1.0
    history_df.loc[history_df['count'] == 1, 'repeat_boost'] = 1.3
    history_df.loc[history_df['count'] == 2, 'repeat_boost'] = 2.0
    history_df.loc[history_df['count'] >= 3, 'repeat_boost'] = 2.5
    
    # 2. Event Type 차등
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
    
    print(f"  ✓ Completed for {result_df['user_id'].nunique():,} users")
    
    return result_df

def build_multi_model_ensemble(model_outputs, weights, score_method='rrf', K=60):
    """
    N개 모델의 출력을 outer merge하여 통합 점수 생성
    
    Args:
        model_outputs: list of (name, DataFrame) - 각 모델의 (user_id, item_id) Top-10
        weights: dict of {name: weight}
        score_method: 'rrf' (Reciprocal Rank Fusion) or 'rank_inv' (1/rank)
        K: RRF constant (default 60)
    
    Returns:
        DataFrame with columns [user_id, item_id, final_score]
    """
    print(f"\nBuilding {len(model_outputs)}-model ensemble...")
    print(f"  Score method: {score_method}, K={K}")
    print(f"  Weights: {weights}")
    
    merged = None
    
    for name, df in model_outputs:
        # Assign rank-based score
        df = df.copy()
        df['rank'] = df.groupby('user_id').cumcount() + 1
        
        if score_method == 'rrf':
            df[f'score_{name}'] = 1.0 / (K + df['rank'])
        else:  # rank_inv
            df[f'score_{name}'] = 1.0 / df['rank']
        
        score_col = f'score_{name}'
        
        if merged is None:
            merged = df[['user_id', 'item_id', score_col]].copy()
        else:
            merged = pd.merge(
                merged,
                df[['user_id', 'item_id', score_col]],
                on=['user_id', 'item_id'],
                how='outer'
            )
        
        print(f"  ✓ {name}: {df['user_id'].nunique():,} users, {len(df):,} rows")
    
    # Fill NaN
    score_cols = [f'score_{name}' for name, _ in model_outputs]
    for col in score_cols:
        merged[col] = merged[col].fillna(0)
    
    # Calculate weighted final score
    merged['final_score'] = 0.0
    for name, _ in model_outputs:
        w = weights.get(name, 0)
        merged['final_score'] += w * merged[f'score_{name}']
    
    total_candidates = len(merged)
    n_users = merged['user_id'].nunique()
    avg_per_user = total_candidates / n_users
    
    print(f"\n  ✓ Total candidates: {total_candidates:,}")
    print(f"  ✓ Users: {n_users:,}")
    print(f"  ✓ Avg candidates/user: {avg_per_user:.1f}")
    
    return merged[['user_id', 'item_id', 'final_score']]

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Ensemble + Phase 1 Boost")
    
    # Model outputs
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv", type=str)
    parser.add_argument("--xgb_output", default="../output/output_reranked_21.csv", type=str)
    parser.add_argument("--catboost_output", default="../output/output_catboost_final_29.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    
    # Ensemble config
    parser.add_argument("--mode", default="3way", choices=['3way', '4way', '3way_catboost'],
                        help="3way=ALS+SASRec+XGB, 4way=+CatBoost, 3way_catboost=ALS+SASRec+CatBoost")
    parser.add_argument("--score_method", default="rrf", choices=['rrf', 'rank_inv'],
                        help="rrf=1/(K+rank), rank_inv=1/rank")
    parser.add_argument("--K", default=60, type=int, help="RRF constant")
    
    # 3-Way weights (기존 triplet에서 검증된 비율 기반)
    parser.add_argument("--w_als", default=0.6, type=float)
    parser.add_argument("--w_sasrec", default=0.25, type=float)
    parser.add_argument("--w_xgb", default=0.15, type=float)
    parser.add_argument("--w_catboost", default=0.15, type=float)
    
    # Output
    parser.add_argument("--output_path", default="../output/output_triplet_boosted.csv", type=str)
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Multi-Model Ensemble + Phase 1 Boost")
    print(f"Mode: {args.mode}")
    print(f"Target: 3-Way(0.1196) + Phase1 boost → 0.1330+ 돌파")
    print("="*60)
    
    # 1. Load model outputs
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    
    model_outputs = [('als', als_df), ('sasrec', sasrec_df)]
    weights = {'als': args.w_als, 'sasrec': args.w_sasrec}
    
    if args.mode in ['3way', '4way']:
        xgb_df = pd.read_csv(args.xgb_output)
        model_outputs.append(('xgb', xgb_df))
        weights['xgb'] = args.w_xgb
        
    if args.mode == '3way_catboost':
        cat_df = pd.read_csv(args.catboost_output)
        model_outputs.append(('catboost', cat_df))
        weights['catboost'] = args.w_catboost
    
    if args.mode == '4way':
        cat_df = pd.read_csv(args.catboost_output)
        model_outputs.append(('catboost', cat_df))
        weights['catboost'] = args.w_catboost
    
    # Normalize weights
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    print(f"\n  Normalized weights: {weights}")
    
    # 2. Build ensemble (wider candidate pool)
    print(f"\n[2/4] Building multi-model ensemble...")
    candidates_df = build_multi_model_ensemble(
        model_outputs, weights, 
        score_method=args.score_method, K=args.K
    )
    
    # 3. Load history & apply Phase 1 boost
    print(f"\n[3/4] Loading enhanced history...")
    history_df = load_enhanced_history(args.train_data)
    
    print(f"\n[4/4] Applying Phase 1 boost...")
    result_df = apply_phase1_boost(candidates_df, history_df)
    
    # 4. Save
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    # Validation
    n_users = result_df['user_id'].nunique()
    n_rows = len(result_df)
    items_per_user = result_df.groupby('user_id').size()
    
    print("\n" + "="*60)
    print(f"✅ {args.mode} + Phase 1 Boost completed!")
    print(f"📊 Total predictions: {n_rows:,}")
    print(f"👥 Users: {n_users:,}")
    print(f"📋 Items/user: min={items_per_user.min()}, max={items_per_user.max()}, mean={items_per_user.mean():.1f}")
    print(f"💾 Saved to: {args.output_path}")
    
    if n_users == 638257 and n_rows == 6382570:
        print(f"✅ Validation PASSED: 638,257 users × 10 items")
    else:
        print(f"⚠️ Validation WARNING: Expected 638,257 users × 10 items")
    
    print("="*60)

if __name__ == "__main__":
    main()
