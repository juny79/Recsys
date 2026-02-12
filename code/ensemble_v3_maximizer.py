"""
V3 Score Maximizer: 5-Phase 체계적 테스트
==========================================
데이터를 1번만 로드하고, Phase A~E 5개 변형을 모두 생성

Base: 4-Way Ensemble + Phase 1+ Enhanced (0.1344)
Phase A: +Category(ultra-conservative)
Phase B: +Category+Brand+PriceGuard
Phase C: +Segment Differentiation
Phase D: +Last Session Category
Phase E: All combined + Conversion Rate

Usage:
  python ensemble_v3_maximizer.py                    # Phase A~E 전부
  python ensemble_v3_maximizer.py --phases A B       # 특정 Phase만
  python ensemble_v3_maximizer.py --phases D          # Phase D만
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
import gc

# ============================================================
# Phase Configurations
# ============================================================
PHASE_CONFIGS = {
    'A': {
        'name': 'Ultra-Conservative (카테고리만)',
        'desc': 'Phase 1+ + cat_top1=1.10',
        'cat_top1': 1.10, 'cat_top2': 1.05,
        'brand_top1': 1.0, 'brand_top2': 1.0,
        'price_guard': False,
        'segment_diff': False,
        'session_cat': False,
        'conversion_boost': False,
        'trending': False,
    },
    'F': {
        'name': 'Session-Only (세션 카테고리만 추가)',
        'desc': 'Phase 1+ + 마지막 세션 카테고리 부스트만 (가장 새로운 시그널)',
        'cat_top1': 1.0, 'cat_top2': 1.0,
        'brand_top1': 1.0, 'brand_top2': 1.0,
        'price_guard': False,
        'segment_diff': False,
        'session_cat': True, 'session_cat_boost': 1.15, 'session_gap_minutes': 5,
        'conversion_boost': False,
        'trending': False,
    },
    'B': {
        'name': 'Conservative Personalization',
        'desc': 'Phase 1+ + cat + brand + price guard',
        'cat_top1': 1.15, 'cat_top2': 1.08,
        'brand_top1': 1.10, 'brand_top2': 1.05,
        'price_guard': True, 'price_guard_mult': 0.85, 'price_sigma': 3.0,
        'segment_diff': False,
        'session_cat': False,
        'conversion_boost': False,
        'trending': False,
    },
    'C': {
        'name': 'Segment-Differentiated',
        'desc': 'Phase B + user segment differentiation',
        'cat_top1': 1.15, 'cat_top2': 1.08,
        'brand_top1': 1.10, 'brand_top2': 1.05,
        'price_guard': True, 'price_guard_mult': 0.85, 'price_sigma': 3.0,
        'segment_diff': True,
        'seg_light_personal': 0.7, 'seg_heavy_personal': 1.3,
        'seg_light_trending': 1.15, 'seg_heavy_trending': 0.9,
        'session_cat': False,
        'conversion_boost': False,
        'trending': True, 'trending_max': 1.12,
    },
    'D': {
        'name': 'Session Context',
        'desc': 'Phase C + last session category boost',
        'cat_top1': 1.15, 'cat_top2': 1.08,
        'brand_top1': 1.10, 'brand_top2': 1.05,
        'price_guard': True, 'price_guard_mult': 0.85, 'price_sigma': 3.0,
        'segment_diff': True,
        'seg_light_personal': 0.7, 'seg_heavy_personal': 1.3,
        'seg_light_trending': 1.15, 'seg_heavy_trending': 0.9,
        'session_cat': True, 'session_cat_boost': 1.20, 'session_gap_minutes': 5,
        'conversion_boost': False,
        'trending': True, 'trending_max': 1.12,
    },
    'E': {
        'name': 'Full Kitchen Sink',
        'desc': 'All strategies combined',
        'cat_top1': 1.15, 'cat_top2': 1.08,
        'brand_top1': 1.10, 'brand_top2': 1.05,
        'price_guard': True, 'price_guard_mult': 0.85, 'price_sigma': 3.0,
        'segment_diff': True,
        'seg_light_personal': 0.7, 'seg_heavy_personal': 1.3,
        'seg_light_trending': 1.15, 'seg_heavy_trending': 0.9,
        'session_cat': True, 'session_cat_boost': 1.20, 'session_gap_minutes': 5,
        'conversion_boost': True, 'conv_boost_mult': 1.12, 'conv_min_views': 10,
        'trending': True, 'trending_max': 1.12,
    },
}


def load_all_data(train_path):
    """모든 컬럼 한번에 로딩 + 전처리"""
    print(f"Loading ALL data from {train_path}...")
    t0 = time.time()
    
    df = pd.read_parquet(train_path)
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    print(f"  Loaded {len(df):,} rows, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    return df


def build_base_history(df):
    """Phase 1+ Enhanced 기본 부스트 히스토리 (검증된 부분)"""
    print("\n[History] Building base Phase 1+ features...")
    t0 = time.time()
    
    count_df = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')
    
    has_purchase = df[df['event_type'] == 'purchase'][['user_id', 'item_id']].drop_duplicates()
    has_purchase['has_purchase'] = True
    has_cart = df[df['event_type'] == 'cart'][['user_id', 'item_id']].drop_duplicates()
    has_cart['has_cart'] = True
    
    last_time = df.groupby(['user_id', 'item_id'])['event_time'].max().reset_index()
    last_time.columns = ['user_id', 'item_id', 'last_time']
    user_last_time = df.groupby('user_id')['event_time'].max().reset_index()
    user_last_time.columns = ['user_id', 'user_last_time']
    
    hist = count_df
    hist = hist.merge(has_purchase, on=['user_id', 'item_id'], how='left')
    hist = hist.merge(has_cart, on=['user_id', 'item_id'], how='left')
    hist = hist.merge(last_time, on=['user_id', 'item_id'], how='left')
    hist = hist.merge(user_last_time, on='user_id', how='left')
    hist['has_purchase'] = hist['has_purchase'].fillna(False)
    hist['has_cart'] = hist['has_cart'].fillna(False)
    hist['hours_since'] = (hist['user_last_time'] - hist['last_time']).dt.total_seconds() / 3600
    
    print(f"  ✓ {len(hist):,} user-item pairs ({time.time()-t0:.1f}s)")
    return hist


def build_user_profiles(df):
    """유저 프로파일: 카테고리, 브랜드, 가격대, 세그먼트"""
    print("\n[Profile] Building user profiles...")
    t0 = time.time()
    
    # Interaction count + segment
    user_counts = df.groupby('user_id').size().reset_index(name='n_events')
    user_counts['segment'] = pd.cut(
        user_counts['n_events'],
        bins=[0, 5, 20, float('inf')],
        labels=['light', 'medium', 'heavy']
    )
    seg_counts = user_counts['segment'].value_counts()
    print(f"  Segments: L={seg_counts.get('light',0):,}, M={seg_counts.get('medium',0):,}, H={seg_counts.get('heavy',0):,}")
    
    # Event + recency weighting
    max_time = df['event_time'].max()
    event_w = df['event_type'].map({'view': 1.0, 'cart': 3.0, 'purchase': 5.0})
    recency_w = np.where(df['event_time'] >= max_time - pd.Timedelta(days=30), 2.0, 1.0)
    total_w = event_w * recency_w
    
    # Category affinity (top-1, top-2)
    cat_valid = df['category_code'].notna()
    if cat_valid.any():
        cat_df = df.loc[cat_valid, ['user_id', 'category_code']].copy()
        cat_df['w'] = total_w[cat_valid].values
        user_cat = cat_df.groupby(['user_id', 'category_code'])['w'].sum().reset_index()
        user_cat['rank'] = user_cat.groupby('user_id')['w'].rank(ascending=False, method='first')
        
        top1_cat = user_cat[user_cat['rank']==1][['user_id','category_code']].rename(columns={'category_code':'top1_category'})
        top2_cat = user_cat[user_cat['rank']==2][['user_id','category_code']].rename(columns={'category_code':'top2_category'})
    else:
        top1_cat = pd.DataFrame(columns=['user_id','top1_category'])
        top2_cat = pd.DataFrame(columns=['user_id','top2_category'])
    
    # Brand loyalty (top-1, top-2, top-3)
    brand_valid = df['brand'].notna()
    if brand_valid.any():
        brand_df = df.loc[brand_valid, ['user_id', 'brand']].copy()
        brand_df['w'] = total_w[brand_valid].values
        user_brand = brand_df.groupby(['user_id', 'brand'])['w'].sum().reset_index()
        user_brand['rank'] = user_brand.groupby('user_id')['w'].rank(ascending=False, method='first')
        
        top1_brand = user_brand[user_brand['rank']==1][['user_id','brand']].rename(columns={'brand':'top1_brand'})
        top2_brand = user_brand[user_brand['rank']==2][['user_id','brand']].rename(columns={'brand':'top2_brand'})
    else:
        top1_brand = pd.DataFrame(columns=['user_id','top1_brand'])
        top2_brand = pd.DataFrame(columns=['user_id','top2_brand'])
    
    # Price affinity
    user_price = df.groupby('user_id')['price'].agg(['mean', 'std']).reset_index()
    user_price.columns = ['user_id', 'avg_price', 'std_price']
    user_price['std_price'] = user_price['std_price'].fillna(0)
    # Avoid zero std (set minimum std)
    user_price['std_price'] = user_price['std_price'].clip(lower=1.0)
    
    # Merge all
    profile = user_counts[['user_id', 'n_events', 'segment']]
    profile = profile.merge(top1_cat, on='user_id', how='left')
    profile = profile.merge(top2_cat, on='user_id', how='left')
    profile = profile.merge(top1_brand, on='user_id', how='left')
    profile = profile.merge(top2_brand, on='user_id', how='left')
    profile = profile.merge(user_price, on='user_id', how='left')
    
    print(f"  ✓ {len(profile):,} user profiles ({time.time()-t0:.1f}s)")
    return profile


def build_item_metadata(df):
    """아이템 메타: 카테고리, 브랜드, 가격, 전환률"""
    print("\n[Meta] Building item metadata...")
    t0 = time.time()
    
    # Latest category/brand/price
    latest = df.sort_values('event_time').groupby('item_id').last()[['category_code', 'brand', 'price']].reset_index()
    
    # Conversion rate
    item_events = df.groupby(['item_id', 'event_type']).size().unstack(fill_value=0)
    if 'purchase' not in item_events.columns:
        item_events['purchase'] = 0
    if 'view' not in item_events.columns:
        item_events['view'] = 0
    
    item_events['total_views'] = item_events.get('view', 0) + item_events.get('cart', 0) + item_events.get('purchase', 0)
    item_events['conversion_rate'] = item_events['purchase'] / item_events['total_views'].clip(lower=1)
    
    conv = item_events[['total_views', 'conversion_rate']].reset_index()
    latest = latest.merge(conv, on='item_id', how='left')
    latest['conversion_rate'] = latest['conversion_rate'].fillna(0)
    
    print(f"  ✓ {len(latest):,} items, avg conversion: {latest['conversion_rate'].mean():.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    return latest


def build_last_session_categories(df, gap_minutes=5):
    """유저의 마지막 세션에서 가장 많이 본 카테고리"""
    print(f"\n[Session] Building last-session categories (gap={gap_minutes}min)...")
    t0 = time.time()
    
    # Sort by user and time
    sorted_df = df[df['category_code'].notna()].sort_values(['user_id', 'event_time'])
    
    # Compute time gaps
    sorted_df['prev_time'] = sorted_df.groupby('user_id')['event_time'].shift(1)
    sorted_df['gap_min'] = (sorted_df['event_time'] - sorted_df['prev_time']).dt.total_seconds() / 60
    
    # Session boundary: gap > threshold
    sorted_df['new_session'] = (sorted_df['gap_min'] > gap_minutes) | sorted_df['gap_min'].isna()
    sorted_df['session_id'] = sorted_df.groupby('user_id')['new_session'].cumsum()
    
    # Last session per user
    last_session = sorted_df.groupby('user_id')['session_id'].max().reset_index()
    last_session.columns = ['user_id', 'last_session_id']
    
    # Items in last session
    last_sess_data = sorted_df.merge(last_session, on='user_id')
    last_sess_data = last_sess_data[last_sess_data['session_id'] == last_sess_data['last_session_id']]
    
    # Most common category in last session
    last_sess_cat = (last_sess_data.groupby(['user_id', 'category_code'])
                     .size().reset_index(name='cnt'))
    last_sess_cat['rank'] = last_sess_cat.groupby('user_id')['cnt'].rank(ascending=False, method='first')
    
    result = last_sess_cat[last_sess_cat['rank']==1][['user_id', 'category_code']].rename(
        columns={'category_code': 'last_session_category'})
    
    print(f"  ✓ {len(result):,} users with last-session category ({time.time()-t0:.1f}s)")
    return result


def compute_trending_items(df, top_n=500):
    """최근 14일 인기 급상승 아이템"""
    print("\n[Trend] Computing trending items...")
    t0 = time.time()
    
    max_time = df['event_time'].max()
    cutoff = max_time - pd.Timedelta(days=14)
    
    recent_pop = df[df['event_time'] >= cutoff].groupby('item_id').size().reset_index(name='recent')
    old_pop = df[df['event_time'] < cutoff].groupby('item_id').size().reset_index(name='old')
    
    pop = recent_pop.merge(old_pop, on='item_id', how='outer').fillna(0)
    
    recent_days = max(1, 14)
    old_days = max(1, (cutoff - df['event_time'].min()).days)
    
    pop['recent_daily'] = pop['recent'] / recent_days
    pop['old_daily'] = pop['old'] / old_days
    pop['trend_score'] = pop['recent_daily'] / (pop['old_daily'] + 1)
    
    pop = pop[pop['recent'] >= 10]
    trending = pop.nlargest(top_n, 'trend_score')
    
    max_score = trending['trend_score'].max()
    trending_map = dict(zip(trending['item_id'], trending['trend_score'] / max_score))
    
    print(f"  ✓ {len(trending_map)} trending items ({time.time()-t0:.1f}s)")
    return trending_map


def build_4way_ensemble(als_df, sas_df, xgb_df, cat_df, weights, K=60):
    """4-Way RRF Ensemble"""
    print(f"\n[Ensemble] 4-Way RRF (K={K})")
    print(f"  W: ALS={weights['als']:.2f} SAS={weights['sasrec']:.2f} XGB={weights['xgb']:.2f} CAT={weights['catboost']:.2f}")
    t0 = time.time()
    
    for name, model_df in [('als', als_df), ('sasrec', sas_df), ('xgb', xgb_df), ('catboost', cat_df)]:
        model_df['rank'] = model_df.groupby('user_id').cumcount() + 1
        model_df[f'score_{name}'] = 1.0 / (K + model_df['rank'])
    
    merged = als_df[['user_id', 'item_id', 'score_als']].merge(
        sas_df[['user_id', 'item_id', 'score_sasrec']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        xgb_df[['user_id', 'item_id', 'score_xgb']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        cat_df[['user_id', 'item_id', 'score_catboost']], on=['user_id', 'item_id'], how='outer'
    )
    
    for col in ['score_als', 'score_sasrec', 'score_xgb', 'score_catboost']:
        merged[col] = merged[col].fillna(0)
    
    merged['final_score'] = (
        weights['als'] * merged['score_als'] +
        weights['sasrec'] * merged['score_sasrec'] +
        weights['xgb'] * merged['score_xgb'] +
        weights['catboost'] * merged['score_catboost']
    )
    
    print(f"  ✓ {len(merged):,} candidates, {len(merged)/merged['user_id'].nunique():.1f} avg/user ({time.time()-t0:.1f}s)")
    return merged[['user_id', 'item_id', 'final_score']]


def apply_boost(candidates_df, history_df, user_profiles, item_meta, 
                last_session_cats, trending_map, config):
    """범용 부스트 함수: config에 따라 다른 Phase 적용"""
    phase_name = config.get('name', 'Unknown')
    print(f"\n{'='*60}")
    print(f"  Applying: {phase_name}")
    print(f"  {config.get('desc', '')}")
    print(f"{'='*60}")
    t0 = time.time()
    
    # ===== [1] Phase 1+ Base Boost (검증됨, 모든 Phase에 적용) =====
    hist = history_df.copy()
    
    hist['repeat_boost'] = 1.0
    hist.loc[hist['count'] == 1, 'repeat_boost'] = 1.4
    hist.loc[hist['count'] == 2, 'repeat_boost'] = 2.1
    hist.loc[hist['count'] >= 3, 'repeat_boost'] = 2.6
    
    hist.loc[hist['has_purchase'], 'repeat_boost'] *= 1.9
    hist.loc[(~hist['has_purchase']) & (hist['has_cart']), 'repeat_boost'] *= 1.5
    
    hist['recency_boost'] = 1.0
    hist.loc[hist['hours_since'] <= 1, 'recency_boost'] = 2.1
    hist.loc[(hist['hours_since'] > 1) & (hist['hours_since'] <= 24), 'recency_boost'] = 1.6
    hist.loc[(hist['hours_since'] > 24) & (hist['hours_since'] <= 72), 'recency_boost'] = 1.3
    
    hist['is_top3'] = ((hist['has_purchase']) & (hist['count'] >= 2)) | (hist['count'] >= 3)
    hist['base_boost'] = hist['repeat_boost'] * hist['recency_boost']
    hist.loc[hist['is_top3'], 'base_boost'] *= 1.6
    
    # Merge base boost
    merged = candidates_df.copy()
    merged = merged.merge(hist[['user_id', 'item_id', 'base_boost']], on=['user_id', 'item_id'], how='left')
    merged['base_boost'] = merged['base_boost'].fillna(1.0)
    
    # ===== [2] Item metadata merge =====
    merged = merged.merge(item_meta[['item_id', 'category_code', 'brand', 'price', 'conversion_rate', 'total_views']], 
                         on='item_id', how='left')
    
    # ===== [3] User profile merge =====
    merged = merged.merge(user_profiles[['user_id', 'segment', 'top1_category', 'top2_category',
                                          'top1_brand', 'top2_brand', 'avg_price', 'std_price']],
                         on='user_id', how='left')
    
    # ===== [4] Category Affinity Boost =====
    cat_top1 = config.get('cat_top1', 1.0)
    cat_top2 = config.get('cat_top2', 1.0)
    
    merged['cat_boost'] = 1.0
    if cat_top1 > 1.0:
        mask1 = (merged['category_code'] == merged['top1_category']) & merged['top1_category'].notna()
        merged.loc[mask1, 'cat_boost'] = cat_top1
        n1 = mask1.sum()
        
        if cat_top2 > 1.0:
            mask2 = ((merged['category_code'] == merged['top2_category']) & 
                     merged['top2_category'].notna() & (merged['cat_boost'] == 1.0))
            merged.loc[mask2, 'cat_boost'] = cat_top2
            n2 = mask2.sum()
        else:
            n2 = 0
        print(f"  [Cat] top1: {n1:,} boosted, top2: {n2:,} boosted")
    
    # ===== [5] Brand Loyalty Boost =====
    brand_top1 = config.get('brand_top1', 1.0)
    brand_top2 = config.get('brand_top2', 1.0)
    
    merged['brand_boost'] = 1.0
    if brand_top1 > 1.0:
        mask_b1 = (merged['brand'] == merged['top1_brand']) & merged['top1_brand'].notna()
        merged.loc[mask_b1, 'brand_boost'] = brand_top1
        n_b1 = mask_b1.sum()
        
        if brand_top2 > 1.0:
            mask_b2 = ((merged['brand'] == merged['top2_brand']) & 
                       merged['top2_brand'].notna() & (merged['brand_boost'] == 1.0))
            merged.loc[mask_b2, 'brand_boost'] = brand_top2
            n_b2 = mask_b2.sum()
        else:
            n_b2 = 0
        print(f"  [Brand] top1: {n_b1:,}, top2: {n_b2:,}")
    
    # ===== [6] Price Guard =====
    merged['price_boost'] = 1.0
    if config.get('price_guard', False):
        sigma = config.get('price_sigma', 3.0)
        penalty = config.get('price_guard_mult', 0.85)
        
        price_diff = np.abs(merged['price'] - merged['avg_price'])
        price_outlier = price_diff > (sigma * merged['std_price'])
        # Only apply penalty where we have valid price data
        valid_price = merged['price'].notna() & merged['avg_price'].notna() & merged['std_price'].notna()
        apply_penalty = price_outlier & valid_price
        
        merged.loc[apply_penalty, 'price_boost'] = penalty
        n_penalized = apply_penalty.sum()
        print(f"  [Price] {n_penalized:,} items penalized (>{sigma}σ deviation)")
    
    # ===== [7] Segment Differentiation =====
    if config.get('segment_diff', False):
        light_mult = config.get('seg_light_personal', 0.7)
        heavy_mult = config.get('seg_heavy_personal', 1.3)
        
        # Adjust personal boost (cat/brand) strength by segment
        for seg, mult in [('light', light_mult), ('heavy', heavy_mult)]:
            mask = merged['segment'] == seg
            if mult != 1.0:
                merged.loc[mask, 'cat_boost'] = 1.0 + (merged.loc[mask, 'cat_boost'] - 1.0) * mult
                merged.loc[mask, 'brand_boost'] = 1.0 + (merged.loc[mask, 'brand_boost'] - 1.0) * mult
            print(f"  [Seg] {seg}: {mask.sum():,} candidates, personal_mult={mult}")
    
    # ===== [8] Last Session Category =====
    merged['session_boost'] = 1.0
    if config.get('session_cat', False) and last_session_cats is not None:
        session_boost_val = config.get('session_cat_boost', 1.20)
        
        merged = merged.merge(last_session_cats, on='user_id', how='left')
        mask_sess = ((merged['category_code'] == merged['last_session_category']) & 
                     merged['last_session_category'].notna())
        merged.loc[mask_sess, 'session_boost'] = session_boost_val
        
        n_sess = mask_sess.sum()
        print(f"  [Session] {n_sess:,} items boosted by last-session category (×{session_boost_val})")
        merged.drop(columns=['last_session_category'], inplace=True, errors='ignore')
    
    # ===== [9] Trending Items =====
    merged['trending_boost'] = 1.0
    if config.get('trending', False) and trending_map:
        trend_max = config.get('trending_max', 1.12)
        
        trend_scores = merged['item_id'].map(trending_map).fillna(0)
        merged['trending_boost'] = 1.0 + trend_scores * (trend_max - 1.0)
        
        # Segment differentiation for trending
        if config.get('segment_diff', False):
            light_trend = config.get('seg_light_trending', 1.15)
            heavy_trend = config.get('seg_heavy_trending', 0.9)
            
            for seg, mult in [('light', light_trend), ('heavy', heavy_trend)]:
                mask = merged['segment'] == seg
                if mult != 1.0:
                    merged.loc[mask, 'trending_boost'] = 1.0 + (merged.loc[mask, 'trending_boost'] - 1.0) * mult
        
        n_trend = (merged['trending_boost'] > 1.001).sum()
        print(f"  [Trend] {n_trend:,} items boosted")
    
    # ===== [10] Conversion Rate Boost =====
    merged['conv_boost'] = 1.0
    if config.get('conversion_boost', False):
        conv_mult = config.get('conv_boost_mult', 1.12)
        min_views = config.get('conv_min_views', 10)
        
        # High conversion items the user viewed but didn't purchase
        # This signal is most useful for items in the candidate list that user has seen
        high_conv = (merged['conversion_rate'] > merged['conversion_rate'].quantile(0.9)) & \
                    (merged['total_views'] >= min_views)
        merged.loc[high_conv, 'conv_boost'] = conv_mult
        
        n_conv = high_conv.sum()
        print(f"  [Conv] {n_conv:,} high-conversion items boosted (×{conv_mult})")
    
    # ===== [FINAL] Composite Score =====
    merged['final_score'] = (
        merged['final_score'] * 
        merged['base_boost'] * 
        merged['cat_boost'] * 
        merged['brand_boost'] * 
        merged['price_boost'] *
        merged['session_boost'] *
        merged['trending_boost'] *
        merged['conv_boost']
    )
    
    # Top-10 per user
    result = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
              .groupby('user_id', sort=False).head(10)[['user_id', 'item_id']])
    
    n_users = result['user_id'].nunique()
    n_rows = len(result)
    valid = (n_users == 638257 and n_rows == 6382570)
    
    print(f"\n  Result: {n_users:,} users, {n_rows:,} rows {'✅' if valid else '⚠️'}")
    print(f"  Time: {time.time()-t0:.1f}s")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="V3 Score Maximizer: Multi-Phase Test")
    
    parser.add_argument("--als_output", default="../output/output.csv")
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv")
    parser.add_argument("--xgb_output", default="../output/output_reranked_21.csv")
    parser.add_argument("--catboost_output", default="../output/output_catboost_final_29.csv")
    parser.add_argument("--train_data", default="../data/train.parquet")
    parser.add_argument("--output_dir", default="../output")
    
    parser.add_argument("--w_als", default=0.5, type=float)
    parser.add_argument("--w_sasrec", default=0.25, type=float)
    parser.add_argument("--w_xgb", default=0.10, type=float)
    parser.add_argument("--w_catboost", default=0.15, type=float)
    parser.add_argument("--K", default=60, type=int)
    
    parser.add_argument("--phases", nargs='+', default=['A', 'B', 'C', 'D', 'E', 'F'],
                       choices=['A', 'B', 'C', 'D', 'E', 'F'],
                       help="Which phases to run (default: all)")
    
    args = parser.parse_args()
    
    phases = [p.upper() for p in args.phases]
    
    print("="*60)
    print(f"V3 Score Maximizer - Phases: {', '.join(phases)}")
    print(f"Base: 4-Way + Phase 1+ Enhanced (0.1344)")
    print("="*60)
    
    total_start = time.time()
    
    # ========== 1. Load model outputs (1회) ==========
    print(f"\n{'='*40}")
    print(f"[STEP 1/5] Loading model outputs...")
    print(f"{'='*40}")
    
    als_df = pd.read_csv(args.als_output)
    sas_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgb_output)
    cat_df = pd.read_csv(args.catboost_output)
    
    print(f"  ALS: {len(als_df):,}, SAS: {len(sas_df):,}, XGB: {len(xgb_df):,}, CAT: {len(cat_df):,}")
    
    # ========== 2. Build ensemble (1회) ==========
    print(f"\n{'='*40}")
    print(f"[STEP 2/5] Building ensemble...")
    print(f"{'='*40}")
    
    weights = {'als': args.w_als, 'sasrec': args.w_sasrec, 'xgb': args.w_xgb, 'catboost': args.w_catboost}
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    
    candidates_df = build_4way_ensemble(als_df, sas_df, xgb_df, cat_df, weights, args.K)
    
    # Free model outputs
    del als_df, sas_df, xgb_df, cat_df
    gc.collect()
    
    # ========== 3. Load & preprocess data (1회) ==========
    print(f"\n{'='*40}")
    print(f"[STEP 3/5] Loading training data & building features...")
    print(f"{'='*40}")
    
    df = load_all_data(args.train_data)
    
    history_df = build_base_history(df)
    user_profiles = build_user_profiles(df)
    item_meta = build_item_metadata(df)
    
    # Conditional: session categories (only if D or E)
    need_session = any(p in phases for p in ['D', 'E'])
    if need_session:
        last_session_cats = build_last_session_categories(df, gap_minutes=5)
    else:
        last_session_cats = None
    
    # Conditional: trending items (only if C, D, or E)
    need_trending = any(p in phases for p in ['C', 'D', 'E'])
    if need_trending:
        trending_map = compute_trending_items(df)
    else:
        trending_map = {}
    
    del df
    gc.collect()
    
    # ========== 4. Run each phase ==========
    print(f"\n{'='*40}")
    print(f"[STEP 4/5] Running {len(phases)} phases...")
    print(f"{'='*40}")
    
    results = {}
    for phase in phases:
        config = PHASE_CONFIGS[phase]
        
        output_path = os.path.join(args.output_dir, f"output_v3_phase{phase}.csv")
        
        result = apply_boost(
            candidates_df, history_df, user_profiles, item_meta,
            last_session_cats, trending_map, config
        )
        
        result.to_csv(output_path, index=False)
        results[phase] = output_path
        print(f"  💾 Saved: {output_path}")
    
    # ========== 5. Summary ==========
    print(f"\n{'='*60}")
    print(f"[STEP 5/5] SUMMARY")
    print(f"{'='*60}")
    
    total_time = time.time() - total_start
    
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"\n  Generated outputs:")
    for phase, path in results.items():
        config = PHASE_CONFIGS[phase]
        print(f"    Phase {phase}: {config['name']}")
        print(f"             {path}")
    
    print(f"\n  Next: Submit each Phase to leaderboard and compare:")
    print(f"    Phase A: 0.1344 → ? (카테고리만)")
    print(f"    Phase B: 0.1344 → ? (+브랜드+가격)")
    print(f"    Phase C: 0.1344 → ? (+세그먼트)")
    print(f"    Phase D: 0.1344 → ? (+세션)")
    print(f"    Phase E: 0.1344 → ? (전부)")
    print(f"\n  ⚡ Best practice: Phase A부터 순차 제출")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
