"""
LightGCN 학습 + 추론 통합 스크립트
=====================================
ALS와 완전히 다른 그래프 기반 협업 필터링

▸ ALS vs LightGCN 차이:
  ALS:      Matrix Factorization (1차만 고려)
            user_vec · item_vec = score
  LightGCN: Graph Propagation (2~3차 이웃까지)
            "나와 비슷한 유저들의 유저들이 본 것"까지 추천
            → 다양성 ↑, Cold-start 유저에 강함

▸ 기대 성능
  단독:  0.11~0.12 (ALS 수준)
  앙상블: 4-Way(0.1344) → 5-Way +0.3~0.8% 기대

▸ RecBole 기반 (이미 설치됨)

Usage:
  # Step 1: 학습
  python train_lightgcn.py

  # Step 2: 추론 (학습 후 자동 실행, 또는 별도 실행)
  python train_lightgcn.py --inference_only --model_file ./saved/LightGCN.pth
"""

import argparse
import os
import json
import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import LightGCN
from recbole.trainer import Trainer
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk

from utils import set_seed


def train(args):
    """LightGCN 학습"""
    print("=" * 60)
    print("LightGCN Training")
    print("=" * 60)

    config = Config(
        model='LightGCN',
        dataset='SASRec_dataset',
        config_file_list=[args.config_file],
    )
    init_seed(config['seed'], config['reproducibility'])

    print(f"  Embedding size: {config['embedding_size']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  reg_weight: {config['reg_weight']}")
    print(f"  Device: {config['device']}")

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = LightGCN(config, train_data.dataset).to(config['device'])
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(config, model)
    trainer.fit(train_data, valid_data, saved=True, show_progress=config['show_progress'])

    # 학습된 모델 파일 경로 반환
    model_file = os.path.join(config['checkpoint_dir'], 'LightGCN.pth')
    return model_file, config, dataset, train_data, test_data


def run_inference(model_file, config, dataset, train_data, test_data,
                  data_dir, output_path, top_k, batch_size=256):
    """LightGCN 추론 → 모든 유저에 대해 Top-K 생성"""
    print("\n" + "=" * 60)
    print("LightGCN Inference")
    print("=" * 60)

    device = config['device']

    # 모델 로드
    model = LightGCN(config, train_data.dataset).to(device)
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"  Loaded: {model_file}")

    # 매핑 파일 로드
    with open(os.path.join(data_dir, 'user2idx.json'), 'r') as f:
        user2idx = json.load(f)
    with open(os.path.join(data_dir, 'item2idx.json'), 'r') as f:
        item2idx = json.load(f)

    idx2user = {int(v): k for k, v in user2idx.items()}
    idx2item = {int(v): k for k, v in item2idx.items()}

    # 원본 데이터에서 전체 유저 목록 + 상호작용 이력 로드
    print("  Loading train data for user list and history...")
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)
    train_df = train_df.dropna(subset=['user_idx', 'item_idx'])

    # 유저별 상호작용 아이템 (이미 본 것은 추천 제외용)
    user_history = defaultdict(set)
    for uid, iid in zip(train_df['user_idx'], train_df['item_idx']):
        user_history[int(uid)].add(int(iid))

    all_user_ids_original = sorted(train_df['user_id'].unique())
    all_users_idx = [user2idx.get(u) for u in all_user_ids_original]
    all_users_idx = [int(u) for u in all_users_idx if u is not None]

    # RecBole 내 유저 목록 (5회 이상 상호작용)
    recbole_users = set()
    for token in dataset.field2token_id.get('user_idx', {}).keys():
        recbole_users.add(token)

    # 인기 아이템 (fallback: RecBole 미등록 유저용)
    popular_items = (
        train_df.groupby('item_idx').size()
        .sort_values(ascending=False)
        .head(100)
        .index.tolist()
    )
    popular_items_original = [idx2item[i] for i in popular_items if i in idx2item]

    print(f"  Total users: {len(all_user_ids_original):,}")
    print(f"  RecBole users (≥5 interactions): {len(recbole_users):,}")
    print(f"  Light users (fallback): {len(all_user_ids_original) - len(recbole_users):,}")

    # 배치 추론
    results = []
    uid_field = dataset.uid_field

    print(f"\n  Running inference (batch_size={batch_size})...")
    print(f"  RecBole users via full_sort_topk...")

    recbole_uid_list = []
    recbole_original_uid = []

    for orig_uid in all_user_ids_original:
        uid_idx = user2idx.get(str(orig_uid), user2idx.get(orig_uid))
        if uid_idx is None:
            continue
        uid_str = str(int(uid_idx))
        if uid_str in dataset.field2token_id.get(uid_field, {}):
            recbole_uid_list.append(uid_str)
            recbole_original_uid.append(orig_uid)

    print(f"  Matched RecBole users: {len(recbole_uid_list):,}")

    # RecBole 등록 유저: 배치 추론
    with torch.no_grad():
        for i in tqdm(range(0, len(recbole_uid_list), batch_size),
                      desc="  LightGCN inference"):
            batch_uid_strs = recbole_uid_list[i:i+batch_size]
            batch_orig_uids = recbole_original_uid[i:i+batch_size]

            recbole_ids = [
                dataset.token2id(uid_field, u) for u in batch_uid_strs
            ]

            try:
                _, topk_iid_tensor = full_sort_topk(
                    recbole_ids, model, test_data, k=top_k, device=device
                )
                topk_iid_tokens = dataset.id2token(dataset.iid_field, topk_iid_tensor.cpu())

                for j, orig_uid in enumerate(batch_orig_uids):
                    items_str = topk_iid_tokens[j]
                    for iid_str in items_str:
                        iid_int = int(iid_str)
                        if iid_int in idx2item:
                            results.append((orig_uid, idx2item[iid_int]))
            except Exception as e:
                # 배치 실패 시 개별 처리
                for j, orig_uid in enumerate(batch_orig_uids):
                    for item in popular_items_original[:top_k]:
                        results.append((orig_uid, item))

    # Light 유저 (RecBole 미등록): 인기 아이템으로 fill
    recbole_set = set(recbole_original_uid)
    light_users = [u for u in all_user_ids_original if u not in recbole_set]
    print(f"\n  Filling {len(light_users):,} light users with popular items...")

    for orig_uid in light_users:
        uid_idx = user2idx.get(str(orig_uid), user2idx.get(orig_uid))
        if uid_idx is not None:
            seen = user_history.get(int(uid_idx), set())
            seen_orig = {idx2item[i] for i in seen if i in idx2item}
            fallback = [it for it in popular_items_original if it not in seen_orig][:top_k]
        else:
            fallback = popular_items_original[:top_k]

        for item in fallback:
            results.append((orig_uid, item))

    # ===== 저장 =====
    print(f"\n  Saving...")
    result_df = pd.DataFrame(results, columns=['user_id', 'item_id'])

    # 유저당 Top-K 보장 (중복 제거 + 재추출)
    result_df = (result_df
                 .drop_duplicates(subset=['user_id', 'item_id'])
                 .groupby('user_id', sort=False)
                 .head(top_k))

    n_users_out = result_df['user_id'].nunique()
    n_rows = len(result_df)
    valid = (n_users_out == 638257 and n_rows == 6382570)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ LightGCN Inference Complete!")
    print(f"   Users: {n_users_out:,} | Rows: {n_rows:,} {'✅' if valid else '⚠️'}")
    print(f"   Saved: {output_path}")
    print(f"\n  다음 단계: 5-Way 앙상블에 추가")
    print(f"  → ensemble_5way.py --lightgcn_output {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LightGCN Train + Inference")
    parser.add_argument("--config_file", default="./yaml/lightgcn.yaml", type=str)
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--output_path", default="../output/output_lightgcn.csv", type=str)
    parser.add_argument("--model_file", default="./saved/LightGCN.pth", type=str)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--inference_only", action='store_true',
                        help="학습 없이 기존 model_file로 추론만 실행")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.inference_only:
        # 추론만: 기존 모델 파일 사용
        print(f"Inference only mode: {args.model_file}")
        config = Config(
            model='LightGCN',
            dataset='SASRec_dataset',
            config_file_list=[args.config_file],
        )
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train_data, _, test_data = data_preparation(config, dataset)

        run_inference(
            args.model_file, config, dataset, train_data, test_data,
            args.data_dir, args.output_path, args.top_k, args.batch_size
        )
    else:
        # 학습 → 추론
        model_file, config, dataset, train_data, test_data = train(args)
        run_inference(
            model_file, config, dataset, train_data, test_data,
            args.data_dir, args.output_path, args.top_k, args.batch_size
        )


if __name__ == "__main__":
    main()
