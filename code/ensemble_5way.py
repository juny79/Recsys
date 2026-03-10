"""
5-Way Ensemble: 4-Way + EASE 또는 LightGCN 추가
=================================================
현재 최고: 4-Way + Phase 1+ Enhanced = 0.1344

▸ 5번째 모델 후보
  EASE:     아이템-아이템 co-occurrence (ALS와 완전히 다른 시그널)
  LightGCN: 그래프 전파 (2~3차 이웃까지 캡처)

▸ 기대 성능
  4-Way → 3-Way가 +0.83%였으므로
  5-Way → 0.1344 + 0.3~0.6% = 0.1348~0.1352 기대

Usage:
  # EASE 사용
  python ensemble_5way.py --fifth_model ease --fifth_output ../output/output_ease_cpu.csv

  # LightGCN 사용
  python ensemble_5way.py --fifth_model lightgcn --fifth_output ../output/output_lightgcn.csv

  # 5-Way 가중치 조정
  python ensemble_5way.py --fifth_model ease --w_fifth 0.15 --w_als 0.45
"""

import pandas as pd
import numpy as np
import argparse
import os
import time


def load_phase1_history(train_path):
    """Phase 1+ Enhanced 히스토리 (검증된 부스트)"""
    print(f"Loading history from {train_path}...")
    t0 = time.time()

    df = pd.read_parquet(train_path, columns=['user_id', 'item_id', 'event_type', 'event_time'])
    df['event_time'] = pd.to_datetime(df['event_time'])

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


def apply_phase1_enhanced_boost(candidates_df, hist):
    """Phase 1+ Enhanced (검증 완료, 0.1341→0.1344 기여)"""
    print("Applying Phase 1+ Enhanced boost...")
    t0 = time.time()

    hist = hist.copy()

    # Repeat boost
    hist['repeat_boost'] = 1.0
    hist.loc[hist['count'] == 1, 'repeat_boost'] = 1.4
    hist.loc[hist['count'] == 2, 'repeat_boost'] = 2.1
    hist.loc[hist['count'] >= 3, 'repeat_boost'] = 2.6

    # Event type
    hist.loc[hist['has_purchase'], 'repeat_boost'] *= 1.9
    hist.loc[(~hist['has_purchase']) & (hist['has_cart']), 'repeat_boost'] *= 1.5

    # Recency
    hist['recency_boost'] = 1.0
    hist.loc[hist['hours_since'] <= 1, 'recency_boost'] = 2.1
    hist.loc[(hist['hours_since'] > 1) & (hist['hours_since'] <= 24), 'recency_boost'] = 1.6
    hist.loc[(hist['hours_since'] > 24) & (hist['hours_since'] <= 72), 'recency_boost'] = 1.3

    # Top-3 super boost
    hist['is_top3'] = ((hist['has_purchase']) & (hist['count'] >= 2)) | (hist['count'] >= 3)
    hist['final_boost'] = hist['repeat_boost'] * hist['recency_boost']
    hist.loc[hist['is_top3'], 'final_boost'] *= 1.6

    # Merge + apply
    merged = candidates_df.merge(
        hist[['user_id', 'item_id', 'final_boost']],
        on=['user_id', 'item_id'], how='left'
    )
    merged['final_boost'] = merged['final_boost'].fillna(1.0)
    merged['final_score'] = merged['final_score'] * merged['final_boost']

    result = (merged.sort_values(['user_id', 'final_score'], ascending=[True, False])
              .groupby('user_id', sort=False)
              .head(10)[['user_id', 'item_id']])

    print(f"  ✓ {result['user_id'].nunique():,} users ({time.time()-t0:.1f}s)")
    return result


def build_5way_ensemble(als_df, sasrec_df, xgb_df, catboost_df, fifth_df,
                        weights, K=60):
    """5-Way RRF Ensemble"""
    w = weights
    print(f"\n[Ensemble] 5-Way RRF (K={K})")
    print(f"  W: ALS={w['als']:.2f} SAS={w['sasrec']:.2f} XGB={w['xgb']:.2f} "
          f"CAT={w['catboost']:.2f} 5th={w['fifth']:.2f}")
    t0 = time.time()

    model_list = [
        ('als', als_df),
        ('sasrec', sasrec_df),
        ('xgb', xgb_df),
        ('catboost', catboost_df),
        ('fifth', fifth_df),
    ]

    for name, df in model_list:
        df['rank'] = df.groupby('user_id').cumcount() + 1
        df[f'score_{name}'] = 1.0 / (K + df['rank'])

    merged = als_df[['user_id', 'item_id', 'score_als']].merge(
        sasrec_df[['user_id', 'item_id', 'score_sasrec']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        xgb_df[['user_id', 'item_id', 'score_xgb']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        catboost_df[['user_id', 'item_id', 'score_catboost']], on=['user_id', 'item_id'], how='outer'
    ).merge(
        fifth_df[['user_id', 'item_id', 'score_fifth']], on=['user_id', 'item_id'], how='outer'
    )

    for col in ['score_als', 'score_sasrec', 'score_xgb', 'score_catboost', 'score_fifth']:
        merged[col] = merged[col].fillna(0)

    merged['final_score'] = (
        w['als'] * merged['score_als'] +
        w['sasrec'] * merged['score_sasrec'] +
        w['xgb'] * merged['score_xgb'] +
        w['catboost'] * merged['score_catboost'] +
        w['fifth'] * merged['score_fifth']
    )

    avg_cand = len(merged) / merged['user_id'].nunique()
    print(f"  ✓ {len(merged):,} candidates, {avg_cand:.1f} avg/user ({time.time()-t0:.1f}s)")
    return merged[['user_id', 'item_id', 'final_score']]


def main():
    parser = argparse.ArgumentParser(description="5-Way Ensemble")

    # 기존 4개 모델
    parser.add_argument("--als_output", default="../output/output.csv")
    parser.add_argument("--sasrec_output", default="../output/output_sasrec_fixed_19.csv")
    parser.add_argument("--xgb_output", default="../output/output_reranked_21.csv")
    parser.add_argument("--catboost_output", default="../output/output_catboost_final_29.csv")
    parser.add_argument("--train_data", default="../data/train.parquet")

    # 5번째 모델
    parser.add_argument("--fifth_model", default="ease",
                        choices=["ease", "lightgcn"],
                        help="ease: 아이템-아이템 CF | lightgcn: 그래프 CF")
    parser.add_argument("--fifth_output", default="../output/output_ease_cpu.csv",
                        help="5번째 모델 출력 파일")

    # 출력
    parser.add_argument("--output_path", default="../output/output_5way.csv")

    # 앙상블 가중치 (합산하면 1.0이 되도록 자동 정규화)
    parser.add_argument("--w_als", default=0.43, type=float)
    parser.add_argument("--w_sasrec", default=0.22, type=float)
    parser.add_argument("--w_xgb", default=0.08, type=float)
    parser.add_argument("--w_catboost", default=0.12, type=float)
    parser.add_argument("--w_fifth", default=0.15, type=float,
                        help="5번째 모델 가중치. EASE/LightGCN: 0.10~0.20 권장")
    parser.add_argument("--K", default=60, type=int)

    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print(f"5-Way Ensemble: 4-Way + {args.fifth_model.upper()}")
    print(f"Base: 4-Way + Phase 1+ Enhanced (0.1344)")
    print(f"Target: 0.1350+")
    print("=" * 60)

    # ===== 1. 모델 출력 로딩 =====
    print(f"\n[1/4] Loading model outputs...")
    als_df = pd.read_csv(args.als_output)
    sasrec_df = pd.read_csv(args.sasrec_output)
    xgb_df = pd.read_csv(args.xgb_output)
    catboost_df = pd.read_csv(args.catboost_output)
    fifth_df = pd.read_csv(args.fifth_output)

    print(f"  ALS:{len(als_df):,} SAS:{len(sasrec_df):,} XGB:{len(xgb_df):,} "
          f"CAT:{len(catboost_df):,} {args.fifth_model.upper()}:{len(fifth_df):,}")

    # 5번째 모델 커버리지 확인
    fifth_users = fifth_df['user_id'].nunique()
    print(f"  {args.fifth_model.upper()} covers: {fifth_users:,} / 638,257 users "
          f"({fifth_users/638257*100:.1f}%)")

    # ===== 2. 앙상블 빌드 =====
    print(f"\n[2/4] Building 5-Way ensemble...")
    weights = {
        'als': args.w_als,
        'sasrec': args.w_sasrec,
        'xgb': args.w_xgb,
        'catboost': args.w_catboost,
        'fifth': args.w_fifth,
    }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    candidates_df = build_5way_ensemble(
        als_df, sasrec_df, xgb_df, catboost_df, fifth_df, weights, args.K
    )

    # ===== 3. 히스토리 로딩 =====
    print(f"\n[3/4] Loading history...")
    hist = load_phase1_history(args.train_data)

    # ===== 4. Phase 1+ 부스트 적용 =====
    print(f"\n[4/4] Applying Phase 1+ Enhanced boost...")
    result = apply_phase1_enhanced_boost(candidates_df, hist)

    # ===== 저장 =====
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_csv(args.output_path, index=False)

    n_users_out = result['user_id'].nunique()
    n_rows = len(result)
    valid = (n_users_out == 638257 and n_rows == 6382570)

    print("\n" + "=" * 60)
    print(f"✅ 5-Way Ensemble Complete!")
    print(f"   Model: 4-Way + {args.fifth_model.upper()}")
    print(f"   Users: {n_users_out:,} | Rows: {n_rows:,} {'✅' if valid else '⚠️'}")
    print(f"   Saved: {args.output_path}")
    print(f"   Total: {(time.time()-total_start)/60:.1f} min")
    print(f"\n   Weights: ALS={weights['als']:.2f} SAS={weights['sasrec']:.2f} "
          f"XGB={weights['xgb']:.2f} CAT={weights['catboost']:.2f} "
          f"{args.fifth_model.upper()}={weights['fifth']:.2f}")
    print(f"\n   3-Way→4-Way improved +0.83%")
    print(f"   4-Way→5-Way expected +0.3~0.6% (if models are diverse)")
    print("=" * 60)


if __name__ == "__main__":
    main()
