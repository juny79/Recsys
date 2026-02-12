"""
nDCG@10 최적화를 위한 Post-Processing 앙상블

전략:
1. Repeat Boost: 사용자의 재방문 아이템 우선 순위 상향 (18.3% 재방문율)
2. Category Consistency: 최근 카테고리와 일치하는 아이템 부스팅 (68.4% 카테고리 일관성)
3. Fine-grained Recency: 시간별 감쇠로 최근 행동에 높은 가중치
4. Diversity Control: Top-10의 카테고리 다양성 유지
"""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tqdm import tqdm

def load_user_history(train_path, target_users=None):
    """사용자 히스토리 로드 및 분석 (필요한 유저만 로딩)"""
    print(f"Loading user history from {train_path}...")
    
    # 필요한 컬럼만 로딩 (메모리 절약)
    required_cols = ['user_id', 'item_id', 'event_time', 'event_type', 'category_code']
    df = pd.read_parquet(train_path, columns=required_cols)
    
    print(f"Total interactions: {len(df):,}")
    print(f"Total unique users: {df['user_id'].nunique():,}")
    
    # 타겟 유저만 필터링 (핵심 최적화!)
    if target_users is not None:
        print(f"Filtering for {len(target_users):,} target users...")
        df = df[df['user_id'].isin(target_users)]
        print(f"✓ Filtered to {len(df):,} interactions ({len(df)/8350311*100:.1f}% of data)")
    
    # 이벤트 시간 파싱
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values(['user_id', 'event_time'])
    
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    user_history = {}
    
    # 그룹화를 통한 효율적인 처리
    print("Processing user histories...")
    for user_id, user_df in tqdm(df.groupby('user_id'), desc="Loading history"):
        # 아이템별 집계
        item_counts = {}
        for item_id, item_df in user_df.groupby('item_id'):
            item_counts[item_id] = {
                ('event_time', 'count'): len(item_df),
                ('event_time', 'max'): item_df['event_time'].max(),
                ('event_type', '<lambda>'): item_df['event_type'].tolist()
            }
        
        # 최근 카테고리 (null이 아닌 것만, 최대 5개)
        recent_cats = user_df[user_df['category_code'].notna()]['category_code'].tail(5).tolist()
        
        user_history[user_id] = {
            'item_counts': item_counts,
            'recent_categories': recent_cats,
            'last_interaction': user_df['event_time'].max(),
            'total_interactions': len(user_df)
        }
    
    print(f"✓ Loaded history for {len(user_history):,} users")
    
    # item_to_category 추출
    print("Extracting item metadata...")
    item_to_category = {}
    for item_id, cat in df[['item_id', 'category_code']].drop_duplicates().values:
        if pd.notna(cat):
            item_to_category[item_id] = cat
    print(f"✓ Loaded metadata for {len(item_to_category):,} items")
    
    # 메모리 정리
    del df
    import gc
    gc.collect()
    
    return user_history, item_to_category

# load_item_metadata 함수는 load_user_history에 통합됨 (메모리 효율성)

def calculate_repeat_boost(item_id, user_history_item):
    """재방문 부스팅 계산"""
    if item_id not in user_history_item['item_counts']:
        return 1.0
    
    item_data = user_history_item['item_counts'][item_id]
    count = item_data[('event_time', 'count')]
    event_types = item_data[('event_type', '<lambda>')]
    
    # 기본 재방문 부스트
    boost = 1.0
    
    if count >= 3:
        boost = 2.5  # 3회 이상 본 아이템: 매우 강한 시그널
    elif count >= 2:
        boost = 2.0  # 2회 본 아이템: 강한 시그널
    else:
        boost = 1.3  # 1회 본 아이템: 약한 부스트
    
    # 구매/장바구니 이력이 있으면 추가 부스트
    if 'purchase' in event_types:
        boost *= 1.5
    elif 'cart' in event_types:
        boost *= 1.3
    
    return boost

def calculate_category_boost(item_id, user_history_item, item_to_category):
    """카테고리 일관성 부스팅"""
    if item_id not in item_to_category:
        return 1.0
    
    item_category = item_to_category[item_id]
    recent_categories = user_history_item['recent_categories']
    
    if not recent_categories:
        return 1.0
    
    # 최근 카테고리와 매칭 정도
    if item_category in recent_categories[:2]:  # 최근 2개 카테고리
        return 1.5
    elif item_category in recent_categories[:4]:  # 최근 4개 카테고리
        return 1.2
    elif item_category in recent_categories:
        return 1.1
    
    return 1.0

def calculate_recency_boost(item_id, user_history_item):
    """시간 기반 부스팅"""
    if item_id not in user_history_item['item_counts']:
        return 1.0
    
    last_interaction_time = user_history_item['item_counts'][item_id][('event_time', 'max')]
    overall_last_time = user_history_item['last_interaction']
    
    # 전체 활동 대비 해당 아이템과의 상호작용 시간
    hours_diff = (overall_last_time - last_interaction_time).total_seconds() / 3600
    
    # 시간 기반 감쇠 (1시간 단위)
    if hours_diff <= 1:
        return 2.0  # 1시간 이내: 매우 최근
    elif hours_diff <= 24:
        return 1.5  # 1일 이내: 최근
    elif hours_diff <= 72:
        return 1.2  # 3일 이내: 비교적 최근
    else:
        recency_boost = max(1.0, 1.5 - (hours_diff / 168))  # 주 단위로 감쇠
        return recency_boost

def apply_post_processing(candidates_df, user_history, item_to_category, args):
    """Post-Processing 로직 적용 (RTX 3070 8GB 최적화)"""
    print("Applying post-processing logic...")
    
    processed_results = []
    unique_users = candidates_df['user_id'].unique()
    
    print(f"Processing {len(unique_users):,} users with {len(candidates_df):,} candidates...")
    
    # 배치 처리로 메모리 효율성 향상
    batch_size = 1000
    num_batches = (len(unique_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(unique_users))
        batch_users = unique_users[start_idx:end_idx]
        
        for user_id in tqdm(batch_users, desc=f"Batch {batch_idx+1}/{num_batches}", leave=False):
            user_candidates = candidates_df[candidates_df['user_id'] == user_id].copy()
            
            if user_id not in user_history:
                # 신규 유저는 그대로 유지
                processed_results.append(user_candidates.head(10)[['user_id', 'item_id']])
                continue
        
        user_hist = user_history[user_id]
        
        # 각 후보에 대해 부스팅 점수 계산
        scores = []
        for _, row in user_candidates.iterrows():
            item_id = row['item_id']
            base_score = row.get('final_score', row.get('score', 1.0))
            
            # 1. Repeat Boost
            repeat_boost = calculate_repeat_boost(item_id, user_hist)
            
            # 2. Category Boost
            category_boost = calculate_category_boost(item_id, user_hist, item_to_category)
            
            # 3. Recency Boost
            recency_boost = calculate_recency_boost(item_id, user_hist)
            
            # 종합 점수
            final_score = base_score * repeat_boost * category_boost * recency_boost
            
            scores.append({
                'user_id': user_id,
                'item_id': item_id,
                'final_score': final_score
            })
        
        # 점수 기준 정렬
        scores_df = pd.DataFrame(scores).sort_values('final_score', ascending=False)
        
        # Diversity Control: 카테고리 다양성 유지
        if args.enforce_diversity:
            final_items = []
            category_counts = defaultdict(int)
            max_per_category = args.max_per_category  # 카테고리당 최대 개수
            
            for _, item_row in scores_df.iterrows():
                item_id = item_row['item_id']
                category = item_to_category.get(item_id, 'unknown')
                
                if category_counts[category] < max_per_category:
                    final_items.append(item_row)
                    category_counts[category] += 1
                
                if len(final_items) >= 10:
                    break
            
            # 10개가 안 되면 나머지 채우기
            if len(final_items) < 10:
                for _, item_row in scores_df.iterrows():
                    if item_row['item_id'] not in [x['item_id'] for x in final_items]:
                        final_items.append(item_row)
                        if len(final_items) >= 10:
                            break
            
            final_df = pd.DataFrame(final_items).head(10)[['user_id', 'item_id']]
        else:
            final_df = scores_df.head(10)[['user_id', 'item_id']]
        
            processed_results.append(final_df)
    
    print("\nConsolidating results...")
    result_df = pd.concat(processed_results, ignore_index=True)
    print(f"✓ Post-processing completed for {len(result_df['user_id'].unique()):,} users")
    
    # 메모리 정리
    import gc
    gc.collect()
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="nDCG@10 Optimized Ensemble with Post-Processing")
    
    # Input files
    parser.add_argument("--als_output", default="../output/output.csv", type=str)
    parser.add_argument("--sasrec_output", default="../output/output_sasrec.csv", type=str)
    parser.add_argument("--train_data", default="../data/train.parquet", type=str)
    
    # Output
    parser.add_argument("--output_path", default="../output/output_optimized_ensemble.csv", type=str)
    
    # Ensemble weights (for base scores)
    parser.add_argument("--w_als", default=0.7, type=float, help="ALS weight")
    parser.add_argument("--w_sasrec", default=0.3, type=float, help="SASRec weight")
    
    # Post-processing options
    parser.add_argument("--enforce_diversity", action='store_true', default=True,
                        help="Enforce category diversity in top-10")
    parser.add_argument("--max_per_category", default=4, type=int, 
                        help="Max items per category in top-10")
    
    # Performance options for RTX 3070 8GB
    parser.add_argument("--batch_size", default=1000, type=int,
                        help="Batch size for processing (adjust for memory)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("nDCG@10 Optimized Ensemble with Post-Processing")
    print("="*60)
    
    # 1. Load model outputs (먼저 로딩!)
    print(f"\n[Step 1/4] Loading model outputs...")
    print(f"  Loading ALS: {args.als_output}")
    if not os.path.exists(args.als_output):
        print(f"Error: {args.als_output} not found.")
        return
    als_df = pd.read_csv(args.als_output)
    print(f"  ✓ ALS: {len(als_df):,} predictions for {als_df['user_id'].nunique():,} users")
    
    print(f"  Loading SASRec: {args.sasrec_output}")
    if not os.path.exists(args.sasrec_output):
        print(f"  Warning: SASRec not found. Using only ALS.")
        merged = als_df.copy()
        merged['rank'] = merged.groupby('user_id').cumcount() + 1
        merged['final_score'] = 1.0 / merged['rank']
        target_users = set(als_df['user_id'].unique())
    else:
        sasrec_df = pd.read_csv(args.sasrec_output)
        print(f"  ✓ SASRec: {len(sasrec_df):,} predictions for {sasrec_df['user_id'].nunique():,} users")
        
        # 2. Base ensemble
        print(f"\n[Step 2/4] Creating base ensemble...")
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
        
        target_users = set(merged['user_id'].unique())
    
    print(f"  ✓ Candidates: {len(merged):,}")
    print(f"  ✓ Target users: {len(target_users):,}")
    
    # 3. Load user history (타겟 유저만!)
    print(f"\n[Step 3/4] Loading user history (ONLY {len(target_users):,} users)...")
    user_history, item_to_category = load_user_history(args.train_data, target_users)
    
    # 4. Apply post-processing
    print(f"\n[Step 4/4] Applying post-processing...")
    result_df = apply_post_processing(merged, user_history, item_to_category, args)
    
    # 5. Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    result_df[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    
    print("\n" + "="*60)
    print(f"✅ Optimized ensemble completed!")
    print(f"📊 Total predictions: {len(result_df):,}")
    print(f"👥 Users: {result_df['user_id'].nunique():,}")
    print(f"💾 Saved to: {args.output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
