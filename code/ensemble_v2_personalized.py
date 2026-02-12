"""
V2 Personalized Ensemble: 4-Way + 개인화 부스트
================================================
기존 Phase 1+ Enhanced (0.1344) 위에 추가:
- 전략 1: 카테고리 친화도 부스트
- 전략 2: 브랜드 충성도 부스트  
- 전략 3: 유저 세그먼트별 차등 전략
- 전략 4: 트렌딩 아이템 부스트

목표: 0.1344 → 0.1380+
"""

import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict

def load_full_history(train_path):
    """모든 컬럼을 포함한 히스토리 로딩"""
    print(f"Loading full history from {train_path}...")
    
    df = pd.read_parquet(train_path)
    df['event_time'] = pd.to_datetime(df['event_time'])
    print(f"  Total interactions: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def build_user_profiles(df):
    """유저별 프로파일 (카테고리, 브랜드, 가격대, 세그먼트)"""
    print("\n[Profile] Building user profiles...")
    
    # ---- 1. 기본 통계 ----
    user_counts = df.groupby('user_id').size().reset_index(name='n_events')
    
    # ---- 2. 카테고리 친화도 ----
    print("  Computing category affinity...")
    # 최근 데이터에 더 가중치 (최근 30일 × 2배)
    max_time = df['event_time'].max()
    df['_recency_w'] = np.where(
        df['event_time'] >= max_time - pd.Timedelta(days=30), 2.0, 1.0
    )
    # cart/purchase에 더 가중치
    df['_event_w'] = df['event_type'].map({'view': 1.0, 'cart': 3.0, 'purchase': 5.0})
    df['_total_w'] = df['_recency_w'] * df['_event_w']
    
    user_cat = df.groupby(['user_id', 'category_code'])['_total_w'].sum().reset_index()
    user_cat.columns = ['user_id', 'category_code', 'cat_score']
    
    # 유저별 카테고리 순위 매기기
    user_cat['cat_rank'] = user_cat.groupby('user_id')['cat_score'].rank(ascending=False, method='first')
    
    # top-1, top-2 카테고리
    top1_cat = user_cat[user_cat['cat_rank'] == 1][['user_id', 'category_code']].rename(
        columns={'category_code': 'top1_category'})
    top2_cat = user_cat[user_cat['cat_rank'] == 2][['user_id', 'category_code']].rename(
        columns={'category_code': 'top2_category'})
    
    # ---- 3. 브랜드 충성도 ----
    print("  Computing brand loyalty...")
    user_brand = df.groupby(['user_id', 'brand'])['_total_w'].sum().reset_index()
    user_brand.columns = ['user_id', 'brand', 'brand_score']
    user_brand['brand_rank'] = user_brand.groupby('user_id')['brand_score'].rank(ascending=False, method='first')
    
    top1_brand = user_brand[user_brand['brand_rank'] == 1][['user_id', 'brand']].rename(
        columns={'brand': 'top1_brand'})
    top2_brand = user_brand[user_brand['brand_rank'] == 2][['user_id', 'brand']].rename(
        columns={'brand': 'top2_brand'})
    top3_brand = user_brand[user_brand['brand_rank'] == 3][['user_id', 'brand']].rename(
        columns={'brand': 'top3_brand'})
    
    # ---- 4. 가격대 ----
    print("  Computing price affinity...")
    user_price = df.groupby('user_id')['price'].agg(['mean', 'std']).reset_index()
    user_price.columns = ['user_id', 'avg_price', 'std_price']
    user_price['std_price'] = user_price['std_price'].fillna(0)
    
    # ---- 5. 세그먼트 ----
    print("  Assigning user segments...")
    user_counts['segment'] = pd.cut(
        user_counts['n_events'],
        bins=[0, 5, 20, float('inf')],
        labels=['light', 'medium', 'heavy']
    )
    seg_counts = user_counts['segment'].value_counts()
    print(f"    Light: {seg_counts.get('light', 0):,}, Medium: {seg_counts.get('medium', 0):,}, Heavy: {seg_counts.get('heavy', 0):,}")
    
    # ---- 6. 모두 병합 ----
    profile = user_counts[['user_id', 'n_events', 'segment']]
    profile = profile.merge(top1_cat, on='user_id', how='left')
    profile = profile.merge(top2_cat, on='user_id', how='left')
    profile = profile.merge(top1_brand, on='user_id', how='left')
    profile = profile.merge(top2_brand, on='user_id', how='left')
    profile = profile.merge(top3_brand, on='user_id', how='left')
    profile = profile.merge(user_price, on='user_id', how='left')
    
    print(f"  ✓ Profiles built for {len(profile):,} users")
    
    # cleanup temp cols
    df.drop(columns=['_recency_w', '_event_w', '_total_w'], inplace=True, errors='ignore')
    
    return profile

def build_item_metadata(df):
    """아이템별 메타데이터 (카테고리, 브랜드, 가격)"""
    print("\n[Meta] Building item metadata...")
    
    # 가장 최근 메타를 사용 (카테고리/브랜드/가격은 변할 수 있으므로)
    latest = df.sort_values('event_time').groupby('item_id').last()[['category_code', 'brand', 'price']].reset_index()
    
    print(f"  ✓ Metadata for {len(latest):,} items")
    return latest

def compute_trending_items(df, top_n=500):
    """트렌딩 아이템 식별"""
    print("\n[Trend] Computing trending items...")
    
    max_time = df['event_time'].max()
    
    # 최근 14일 vs 그 이전
    recent = df[df['event_time'] >= max_time - pd.Timedelta(days=14)]
    old = df[df['event_time'] < max_time - pd.Timedelta(days=14)]
    
    recent_pop = recent.groupby('item_id').size().reset_index(name='recent_count')
    old_pop = old.groupby('item_id').size().reset_index(name='old_count')
    
    pop = recent_pop.merge(old_pop, on='item_id', how='outer').fillna(0)
    
    # 최근 기간이 짧으므로 일평균으로 정규화
    recent_days = max(1, (max_time - (max_time - pd.Timedelta(days=14))).days)
    old_days = max(1, ((max_time - pd.Timedelta(days=14)) - df['event_time'].min()).days)
    
    pop['recent_daily'] = pop['recent_count'] / recent_days
    pop['old_daily'] = pop['old_count'] / old_days
    
    # 트렌딩 점수 = 최근 일평균 / (과거 일평균 + 1)
    pop['trending_score'] = pop['recent_daily'] / (pop['old_daily'] + 1)
    
    # 최소 조건: 최근 10건 이상
    pop = pop[pop['recent_count'] >= 10]
    trending = pop.nlargest(top_n, 'trending_score')
    
    trending_items = set(trending['item_id'].values)
    
    # 트렌딩 점수 맵 (정규화)
    max_score = trending['trending_score'].max()
    trending_map = dict(zip(
        trending['item_id'],
        trending['trending_score'] / max_score
    ))
    
    print(f"  ✓ {len(trending_items)} trending items identified")
    print(f"  Top trending score range: {trending['trending_score'].min():.2f} ~ {trending['trending_score'].max():.2f}")
    
    return trending_map

def load_basic_history(df):
    """기본 Phase 1+ 부스트용 히스토리"""
    print("\n[History] Building interaction history...")
    
    # 1. 카운트
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    # 2. 이벤트 타입
    has_purchase = df[df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
    has_purchase['has_purchase'] = True
    
    has_cart = df[df['event_type'] == 'cart'][['user_id', 'item_id']].drop_duplicates()
    has_cart['has_cart'] = True
    
    # 3. 최근 시간
    last_time = df.groupby(['user_id', 'item_id'])['event_time'].max().reset_index()
    last_time.columns = ['user_id', 'item_id', 'last_time']
    
    user_last_time = df.groupby('user_id')['event_time'].max().reset_index()
    user_last_time.columns = ['user_id', 'user_last_time']
    
    # 4. 병합
    hist_df = count_df
    hist_df = hist_df.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist_df = hist_df.merge(user_last_time, on='user_id', how='left')
    
    hist_df['has_purchase'] = hist_df['has_purchase'].fillna(False).astype(bool)
    hist_df['has_cart'] = hist_df['has_cart'].fillna(False).astype(bool)
    hist_df['hours_since'] = (hist_df['user_last_time'] - hist_df['last_time']).dt.total_seconds() / 3600
    
    print(f"  ✓ {len(hist_df):,} user-item pairs")
    
    return hist_df

def build_4way_ensemble(als_df, sasrec_df, xgb_df, catboost_df, weights, K=60):
    """4-Way 앙상블"""
    print(f"\n[Ensemble] Building 4-way ensemble...")
    print(f"  Weights: ALS={weights['als']:.2f}, SASRec={weights['sasrec']:.2f}, XGB={weights['xgb']:.2f}, CatBoost={weights['catboost']:.2f}")
    
    for name, model_df in [('als', als_df), ('sasrec', sasrec_df), ('xgb', xgb_df), ('catboost', catboost_df)]:
        model_df['rank'] = model_df.groupby('user_id').cumcount() + 1
        model_df[f'score_{name}'] = 1.0 / (K + model_df['rank'])
    
    merged = als_df[['user_id', 'item_id', 'score_als']].merge(
        sasrec_df[['user_id', 'item_id', 'score_sasrec']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        xgb_df[['user_id', 'item_id', 'score_xgb']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        catboost_df[['user_id', 'item_id', 'score_catboost']], on=['user_id', 'item_id'], how='outer'
    )
    
    for col in ['score_als', 'score_sasrec', 'score_xgb', 'score_catboost']:
        merged[col] = merged[col].fillna(0)
    
    merged['final_score'] = (
        weights['als'] * merged['score_als'] +
        weights['sasrec'] * merged['score_sasrec'] +
        weights['xgb'] * merged['score_xgb'] +
        weights['catboost'] * merged['score_catboost']
    )
    
    print(f"  ✓ {len(merged):,} candidates ({len(merged)/merged['user_id'].nunique():.1f} avg/user)")
    
    return merged[['user_id', 'item_id', 'final_score']]

def apply_v2_personalized_boost(candidates_df, history_df, user_profiles, item_meta, trending_map, config):
    """V2 개인화 부스트: Phase 1+ + Category + Brand + Segment + Trending"""
    print(f"\n[V2 Boost] Applying personalized boosts...")
    
    # ======= Phase 1+ 기본 부스트 (검증된 것) =======
    print("  [1/5] Phase 1+ base boost...")
    hist = history_df.copy()
    
    # Repeat boost
    hist['repeat_boost'] = 1.0
    hist.loc[hist['count'] == 1, 'repeat_boost'] = config.get('repeat_1', 1.4)
    hist.loc[hist['count'] == 2, 'repeat_boost'] = config.get('repeat_2', 2.1)
    hist.loc[hist['count'] >= 3, 'repeat_boost'] = config.get('repeat_3', 2.6)
    
    # Event type
    hist.loc[hist['has_purchase'], 'repeat_boost'] *= config.get('purchase_mult', 1.9)
    hist.loc[(~hist['has_purchase']) & (hist['has_cart']), 'repeat_boost'] *= config.get('cart_mult', 1.5)
    
    # Recency
    hist['recency_boost'] = 1.0
    hist.loc[hist['hours_since'] <= 1, 'recency_boost'] = config.get('recency_1h', 2.1)
    hist.loc[(hist['hours_since'] > 1) & (hist['hours_since'] <= 24), 'recency_boost'] = config.get('recency_24h', 1.6)
    hist.loc[(hist['hours_since'] > 24) & (hist['hours_since'] <= 72), 'recency_boost'] = config.get('recency_72h', 1.3)
    
    # Top-3
    hist['is_top3'] = ((hist['has_purchase']) & (hist['count'] >= 2)) | (hist['count'] >= 3)
    hist['base_boost'] = hist['repeat_boost'] * hist['recency_boost']
    hist.loc[hist['is_top3'], 'base_boost'] *= config.get('top3_mult', 1.6)
    
    # 후보와 병합
    merged = candidates_df.merge(
        hist[['user_id', 'item_id', 'base_boost']],
        on=['user_id', 'item_id'], how='left'
    )
    merged['base_boost'] = merged['base_boost'].fillna(1.0)
    
    # ======= 아이템 메타데이터 병합 =======
    merged = merged.merge(item_meta, on='item_id', how='left')
    
    # ======= 유저 프로파일 병합 =======
    merged = merged.merge(
        user_profiles[['user_id', 'segment', 'top1_category', 'top2_category', 
                       'top1_brand', 'top2_brand', 'top3_brand', 'avg_price', 'std_price']],
        on='user_id', how='left'
    )
    
    # ======= 전략 1: 카테고리 친화도 =======
    print("  [2/5] Category affinity boost...")
    cat_boost_1 = config.get('cat_top1', 1.4)
    cat_boost_2 = config.get('cat_top2', 1.2)
    
    merged['cat_boost'] = 1.0
    # top-1 카테고리 매칭
    mask_cat1 = (merged['category_code'] == merged['top1_category']) & merged['top1_category'].notna()
    merged.loc[mask_cat1, 'cat_boost'] = cat_boost_1
    # top-2 카테고리 매칭
    mask_cat2 = (merged['category_code'] == merged['top2_category']) & merged['top2_category'].notna() & (merged['cat_boost'] == 1.0)
    merged.loc[mask_cat2, 'cat_boost'] = cat_boost_2
    
    cat_boosted = (merged['cat_boost'] > 1.0).sum()
    print(f"    Boosted {cat_boosted:,} candidates ({cat_boosted/len(merged)*100:.1f}%)")
    
    # ======= 전략 2: 브랜드 충성도 =======
    print("  [3/5] Brand loyalty boost...")
    brand_boost_1 = config.get('brand_top1', 1.25)
    brand_boost_2 = config.get('brand_top2', 1.1)
    
    merged['brand_boost'] = 1.0
    mask_b1 = (merged['brand'] == merged['top1_brand']) & merged['top1_brand'].notna()
    merged.loc[mask_b1, 'brand_boost'] = brand_boost_1
    mask_b2 = (merged['brand'] == merged['top2_brand']) & merged['top2_brand'].notna() & (merged['brand_boost'] == 1.0)
    merged.loc[mask_b2, 'brand_boost'] = brand_boost_2
    mask_b3 = (merged['brand'] == merged['top3_brand']) & merged['top3_brand'].notna() & (merged['brand_boost'] == 1.0)
    merged.loc[mask_b3, 'brand_boost'] = brand_boost_2
    
    brand_boosted = (merged['brand_boost'] > 1.0).sum()
    print(f"    Boosted {brand_boosted:,} candidates ({brand_boosted/len(merged)*100:.1f}%)")
    
    # ======= 전략 3: 유저 세그먼트별 차등 =======
    print("  [4/5] Segment-based differentiation...")
    # Light 유저: 개인 부스트 약화, 인기도/트렌딩 강화
    # Heavy 유저: 개인 부스트(카테고리/브랜드) 강화
    
    seg_personal_mult = {
        'light': config.get('seg_light_personal', 0.7),   # 개인 부스트 약화
        'medium': config.get('seg_med_personal', 1.0),     # 표준
        'heavy': config.get('seg_heavy_personal', 1.3),    # 개인 부스트 강화
    }
    seg_popular_mult = {
        'light': config.get('seg_light_popular', 1.3),     # 인기도 강화
        'medium': config.get('seg_med_popular', 1.0),      # 표준
        'heavy': config.get('seg_heavy_popular', 0.8),     # 인기도 약화
    }
    
    # 세그먼트별 개인 부스트 조정
    for seg, mult in seg_personal_mult.items():
        mask = merged['segment'] == seg
        n = mask.sum()
        if mult != 1.0:
            # 카테고리/브랜드 부스트를 세그먼트에 맞게 조정
            # (boost - 1.0) * mult + 1.0 로 부스트 강도 조절
            merged.loc[mask, 'cat_boost'] = 1.0 + (merged.loc[mask, 'cat_boost'] - 1.0) * mult
            merged.loc[mask, 'brand_boost'] = 1.0 + (merged.loc[mask, 'brand_boost'] - 1.0) * mult
        print(f"    {seg}: {n:,} candidates, personal_mult={mult}, popular_mult={seg_popular_mult[seg]}")
    
    # ======= 전략 4: 트렌딩 아이템 =======
    print("  [5/5] Trending item boost...")
    trending_boost_max = config.get('trending_max', 1.25)
    
    merged['trending_boost'] = merged['item_id'].map(trending_map).fillna(0)
    merged['trending_boost'] = 1.0 + merged['trending_boost'] * (trending_boost_max - 1.0)
    
    # Light 유저에게 트렌딩 부스트 강화
    for seg, mult in seg_popular_mult.items():
        mask = merged['segment'] == seg
        if mult != 1.0:
            merged.loc[mask, 'trending_boost'] = 1.0 + (merged.loc[mask, 'trending_boost'] - 1.0) * mult
    
    trending_boosted = (merged['trending_boost'] > 1.01).sum()
    print(f"    Boosted {trending_boosted:,} candidates ({trending_boosted/len(merged)*100:.1f}%)")
    
    # ======= 최종 점수 계산 =======
    merged['final_score'] = (
        merged['final_score'] *
        merged['base_boost'] *
        merged['cat_boost'] *
        merged['brand_boost'] *
        merged['trending_boost']
    )
    
    # Top-10 선택
    result_df = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
                 .groupby('user_id', sort=False)
                 .head(10)
                 [['user_id', 'item_id']])
    
    print(f"\n  ✓ Final: {result_df['user_id'].nunique():,} users, {len(result_df):,} predictions")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="V2 Personalized Ensemble")
    
    # 모델 경로
    parser.add_argument("--als_output", default="../output/output.csv")
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv")
    parser.add_argument("--xgb_output", default="../output/output_reranked_21.csv")
    parser.add_argument("--catboost_output", default="../output/output_catboost_final_29.csv")
    parser.add_argument("--train_data", default="../data/train.parquet")
    parser.add_argument("--output_path", default="../output/output_v2_personalized.csv")
    
    # 앙상블 가중치
    parser.add_argument("--w_als", default=0.5, type=float)
    parser.add_argument("--w_sasrec", default=0.25, type=float)
    parser.add_argument("--w_xgb", default=0.10, type=float)
    parser.add_argument("--w_catboost", default=0.15, type=float)
    parser.add_argument("--K", default=60, type=int)
    
    # 부스트 파라미터 (프리셋)
    parser.add_argument("--preset", default="balanced", choices=["balanced", "conservative", "aggressive", "category_focus"])
    
    args = parser.parse_args()
    
    # 프리셋 설정
    presets = {
        'balanced': {
            # Phase 1+ (검증됨)
            'repeat_1': 1.4, 'repeat_2': 2.1, 'repeat_3': 2.6,
            'purchase_mult': 1.9, 'cart_mult': 1.5,
            'recency_1h': 2.1, 'recency_24h': 1.6, 'recency_72h': 1.3,
            'top3_mult': 1.6,
            # 카테고리 (신규)
            'cat_top1': 1.4, 'cat_top2': 1.2,
            # 브랜드 (신규)
            'brand_top1': 1.25, 'brand_top2': 1.1,
            # 세그먼트 (신규)
            'seg_light_personal': 0.7, 'seg_med_personal': 1.0, 'seg_heavy_personal': 1.3,
            'seg_light_popular': 1.3, 'seg_med_popular': 1.0, 'seg_heavy_popular': 0.8,
            # 트렌딩 (신규)
            'trending_max': 1.25,
        },
        'conservative': {
            'repeat_1': 1.4, 'repeat_2': 2.1, 'repeat_3': 2.6,
            'purchase_mult': 1.9, 'cart_mult': 1.5,
            'recency_1h': 2.1, 'recency_24h': 1.6, 'recency_72h': 1.3,
            'top3_mult': 1.6,
            'cat_top1': 1.25, 'cat_top2': 1.1,
            'brand_top1': 1.15, 'brand_top2': 1.05,
            'seg_light_personal': 0.8, 'seg_med_personal': 1.0, 'seg_heavy_personal': 1.2,
            'seg_light_popular': 1.2, 'seg_med_popular': 1.0, 'seg_heavy_popular': 0.9,
            'trending_max': 1.15,
        },
        'aggressive': {
            'repeat_1': 1.4, 'repeat_2': 2.1, 'repeat_3': 2.6,
            'purchase_mult': 1.9, 'cart_mult': 1.5,
            'recency_1h': 2.1, 'recency_24h': 1.6, 'recency_72h': 1.3,
            'top3_mult': 1.6,
            'cat_top1': 1.6, 'cat_top2': 1.3,
            'brand_top1': 1.35, 'brand_top2': 1.15,
            'seg_light_personal': 0.5, 'seg_med_personal': 1.0, 'seg_heavy_personal': 1.5,
            'seg_light_popular': 1.5, 'seg_med_popular': 1.0, 'seg_heavy_popular': 0.7,
            'trending_max': 1.35,
        },
        'category_focus': {
            'repeat_1': 1.4, 'repeat_2': 2.1, 'repeat_3': 2.6,
            'purchase_mult': 1.9, 'cart_mult': 1.5,
            'recency_1h': 2.1, 'recency_24h': 1.6, 'recency_72h': 1.3,
            'top3_mult': 1.6,
            'cat_top1': 1.6, 'cat_top2': 1.35,
            'brand_top1': 1.15, 'brand_top2': 1.05,
            'seg_light_personal': 0.6, 'seg_med_personal': 1.0, 'seg_heavy_personal': 1.4,
            'seg_light_popular': 1.2, 'seg_med_popular': 1.0, 'seg_heavy_popular': 0.8,
            'trending_max': 1.2,
        },
    }
    
    config = presets[args.preset]
    
    print("="*60)
    print(f"V2 Personalized Ensemble (preset: {args.preset})")
    print(f"Base: 4-Way + Phase 1+ Enhanced (0.1344)")
    print(f"New: +Category +Brand +Segment +Trending")
    print(f"Target: 0.1380+")
    print("="*60)
    
    # 1. Load model outputs
    print(f"\n[1/6] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgb_output)
    catboost_df = pd.read_csv(args.catboost_output)
    
    # 2. Build ensemble
    weights = {
        'als': args.w_als, 'sasrec': args.w_sasrec,
        'xgb': args.w_xgb, 'catboost': args.w_catboost
    }
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    
    print(f"\n[2/6] Building ensemble...")
    candidates_df = build_4way_ensemble(als_df, sasrec_df, xgb_df, catboost_df, weights, args.K)
    
    # 3. Load full data
    print(f"\n[3/6] Loading training data...")
    df = load_full_history(args.train_data)
    
    # 4. Build profiles & metadata
    print(f"\n[4/6] Building profiles...")
    user_profiles = build_user_profiles(df)
    item_meta = build_item_metadata(df)
    trending_map = compute_trending_items(df)
    history_df = load_basic_history(df)
    
    # 5. Apply V2 boost
    print(f"\n[5/6] Applying V2 boosts...")
    result_df = apply_v2_personalized_boost(
        candidates_df, history_df, user_profiles, item_meta, trending_map, config
    )
    
    # 6. Save
    print(f"\n[6/6] Saving...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df.to_csv(args.output_path, index=False)
    
    n_users = result_df['user_id'].nunique()
    n_rows = len(result_df)
    
    print("\n" + "="*60)
    print(f"✅ V2 Personalized Ensemble completed!")
    print(f"📊 Predictions: {n_rows:,} | Users: {n_users:,}")
    print(f"💾 Saved to: {args.output_path}")
    print(f"🎯 Preset: {args.preset}")
    print(f"📋 Config: cat={config['cat_top1']}/{config['cat_top2']}, brand={config['brand_top1']}/{config['brand_top2']}")
    print(f"   segment: light={config['seg_light_personal']}, heavy={config['seg_heavy_personal']}")
    print(f"   trending={config['trending_max']}")
    
    if n_users == 638257 and n_rows == 6382570:
        print(f"✅ Validation PASSED")
    else:
        print(f"⚠️ Validation: users={n_users}, rows={n_rows}")
    
    print("="*60)

if __name__ == "__main__":
    main()
