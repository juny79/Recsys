"""
EASE (Embarrassingly Shallow AutoEncoder) - CPU 버전
======================================================
기존 train_ease.py가 GPU OOM으로 실패한 문제 수정

▸ 실패 원인
  - 29,502 × 29,502 dense matrix를 GPU에 올림
  - G(3.5GB) + G^-1(3.5GB) = 7GB → RTX 3070 8GB OOM

▸ 수정 방법
  - Gram matrix 계산: scipy sparse (CPU)
  - 역행렬 계산: numpy (float32, ~3.5GB RAM, 가능)
  - 추론: 배치 행렬곱 (numpy)
  - RAM 기준 동작: 시스템 RAM 8GB+ 필요 (보통 16GB OK)

▸ 왜 5번째 모델로 적합한가?
  ALS:     유저-아이템 잠재벡터 (user embedding × item embedding)
  EASE:    순수 아이템-아이템 co-occurrence (매트릭스 직접 분석)
  → 완전히 다른 시그널 → 앙상블 다양성 ↑

Usage:
  python train_ease_cpu.py                     # 기본 실행
  python train_ease_cpu.py --l2_reg 200        # L2 더 작게 (recall 우선)
  python train_ease_cpu.py --l2_reg 1000       # L2 더 크게 (precision 우선)
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--output_path", default="../output/output_ease_cpu.csv", type=str)
    parser.add_argument("--l2_reg", default=500.0, type=float,
                        help="L2 Regularization lambda. 낮을수록 recall↑, 높을수록 precision↑")
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--batch_size", default=2048, type=int,
                        help="추론 배치 크기. RAM 부족 시 줄이세요")
    parser.add_argument("--event_purchase", default=5.0, type=float)
    parser.add_argument("--event_cart", default=3.0, type=float)
    parser.add_argument("--event_view", default=1.0, type=float)
    args = parser.parse_args()

    total_start = time.time()

    # ===== 1. 데이터 로딩 =====
    print("=" * 60)
    print(f"EASE CPU - L2={args.l2_reg}")
    print("=" * 60)
    print(f"\n[1/5] Loading data...")
    t0 = time.time()

    df = pd.read_parquet(args.train_file)
    print(f"  Interactions: {len(df):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ID 매핑
    all_users = sorted(df['user_id'].unique())
    all_items = sorted(df['item_id'].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {i: j for j, i in enumerate(all_items)}
    idx2user = np.array(all_users)
    idx2item = np.array(all_items)

    n_users = len(all_users)
    n_items = len(all_items)
    print(f"  Users: {n_users:,}, Items: {n_items:,}")

    # ===== 2. 상호작용 행렬 구성 =====
    print(f"\n[2/5] Building interaction matrix...")
    t0 = time.time()

    event_w = {
        'purchase': args.event_purchase,
        'cart': args.event_cart,
        'view': args.event_view,
    }
    df['weight'] = df['event_type'].map(event_w).fillna(1.0)

    # 유저-아이템 쌍별 최대 이벤트 가중치 사용
    agg = df.groupby(['user_id', 'item_id'])['weight'].max().reset_index()
    rows = agg['user_id'].map(user2idx).values
    cols = agg['item_id'].map(item2idx).values
    data = agg['weight'].values.astype(np.float32)

    X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    print(f"  Matrix: {n_users:,} × {n_items:,}, nnz={X.nnz:,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== 3. EASE 학습 (CPU) =====
    print(f"\n[3/5] Training EASE (CPU)...")
    print(f"  Computing Gram matrix G = X^T X ...")
    t0 = time.time()

    # G: (n_items × n_items) --- sparse → dense float32
    # 29502 × 29502 × 4 bytes ≈ 3.48 GB
    G = (X.T @ X).toarray().astype(np.float32)
    print(f"  G shape: {G.shape}, dtype: {G.dtype}")
    print(f"  RAM for G: ~{G.nbytes / 1e9:.2f} GB")
    print(f"  Gram time: {time.time()-t0:.1f}s")

    # 대각선에 L2 정규화 추가
    print(f"  Adding L2={args.l2_reg} to diagonal...")
    np.fill_diagonal(G, G.diagonal() + args.l2_reg)

    # G의 역행렬 계산 (= P)
    # Cholesky 방식: G는 Symmetric Positive Definite → LU보다 2배 빠름
    # float32로 계산 (float64 대비 메모리 절반)
    print(f"  Inverting G via Cholesky decomposition...")
    print(f"  ⏱  예상 시간: 5~20분 (CPU 성능에 따라 다름)")
    print(f"  💾 RAM 필요량: ~{G.nbytes * 2 / 1e9:.1f} GB (G + P)")
    t0 = time.time()
    try:
        # Cholesky factorize → solve with identity → G^{-1}
        P = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(G),
            np.eye(n_items, dtype=np.float32)
        ).astype(np.float32)
    except np.linalg.LinAlgError:
        # fallback: standard inversion
        print("  Cholesky failed (ill-conditioned), falling back to numpy.linalg.inv...")
        P = np.linalg.inv(G).astype(np.float32)
    del G
    gc.collect()
    print(f"  Inversion time: {time.time()-t0:.1f}s")
    print(f"  RAM for P: ~{P.nbytes / 1e9:.2f} GB")

    # B = P / (-diag(P)), B_ii = 0
    # In-place 연산으로 RAM 절약 (P와 B 동시 보유 방지)
    print(f"  Computing B = P / -diag(P) (in-place)...")
    t0 = time.time()
    diag_P = P.diagonal().copy()           # (n_items,)
    P /= -diag_P[np.newaxis, :]            # P → B in-place (column-wise division)
    B = P                                  # rename (no copy)
    del P                                  # reference 제거
    np.fill_diagonal(B, 0.0)              # 자기 자신 추천 금지
    gc.collect()
    print(f"  B shape: {B.shape}, RAM: ~{B.nbytes / 1e9:.2f} GB")
    print(f"  B build time: {time.time()-t0:.1f}s")

    # ===== 4. 추론 (Sparse 배치 행렬곱) =====
    # ⚠️ X를 전체 dense로 변환하면 638k × 29.5k × 4B ≈ 75GB → 불가
    # Sparse CSR 방식으로 배치별 처리 (RAM 절약)
    print(f"\n[4/5] Generating predictions (sparse batch, size={args.batch_size})...")
    t0 = time.time()

    all_user_ids = []
    all_item_ids = []

    for start in tqdm(range(0, n_users, args.batch_size)):
        end = min(start + args.batch_size, n_users)
        batch_len = end - start

        # Sparse 슬라이스: (batch, n_items) — 메모리 효율적
        X_batch_sparse = X[start:end]                   # CSR sparse
        S_batch = (X_batch_sparse @ B).astype(np.float32)  # dense (batch, n_items)

        # 이미 상호작용한 아이템 마스킹 (sparse nonzero 활용)
        seen_rows, seen_cols = X_batch_sparse.nonzero()
        S_batch[seen_rows, seen_cols] = -np.inf

        # Top-K
        top_idx = np.argpartition(S_batch, -args.top_k, axis=1)[:, -args.top_k:]
        top_scores = S_batch[np.arange(batch_len)[:, None], top_idx]
        order = np.argsort(-top_scores, axis=1)
        top_idx = top_idx[np.arange(batch_len)[:, None], order]

        # user_id / item_id 매핑
        batch_user_ids = np.repeat(idx2user[start:end], args.top_k)
        batch_item_ids = idx2item[top_idx.flatten()]

        all_user_ids.append(batch_user_ids)
        all_item_ids.append(batch_item_ids)

    print(f"  Prediction time: {time.time()-t0:.1f}s")

    # ===== 5. 저장 =====
    print(f"\n[5/5] Saving...")
    t0 = time.time()

    result = pd.DataFrame({
        'user_id': np.concatenate(all_user_ids),
        'item_id': np.concatenate(all_item_ids),
    })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    result.to_csv(args.output_path, index=False)

    n_users_out = result['user_id'].nunique()
    n_rows = len(result)
    valid = (n_users_out == 638257 and n_rows == 6382570)

    print(f"\n{'='*60}")
    print(f"✅ EASE CPU Training Complete!")
    print(f"   Users: {n_users_out:,} | Rows: {n_rows:,} {'✅' if valid else '⚠️'}")
    print(f"   Saved: {args.output_path}")
    print(f"   Total time: {(time.time()-total_start)/60:.1f} min")
    print(f"{'='*60}")
    print(f"\n  다음 단계: 5-Way 앙상블에 추가")
    print(f"  ensemble_v3_maximizer.py --phases A --ease_output {args.output_path}")


if __name__ == "__main__":
    main()
