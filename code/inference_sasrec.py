import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import argparse
import json
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.utils.case_study import full_sort_topk
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--model_file", default="./saved/SASRec.pth", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--output_file", default="output_sasrec_fixed.csv", type=str)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load configuration
    config = Config(model='SASRec', dataset='SASRec_dataset', config_file_list=['./yaml/sasrec.yaml'])
    config['device'] = device
    
    # Load dataset
    dataset = create_dataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    
    # Load model
    model = SASRec(config, train_data.dataset).to(device)
    checkpoint = torch.load(args.model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print('Data and model load complete')

    # [FIX] 저장된 매핑 파일 사용 - 학습과 동일한 매핑 보장
    with open(os.path.join(args.data_dir,'user2idx.json'),"r") as f:
        user2idx = json.load(f)
    with open(os.path.join(args.data_dir,'item2idx.json'),"r") as f:
        item2idx = json.load(f)
    
    # [FIX] 역매핑도 저장된 파일에서 로드 (또는 user2idx/item2idx에서 생성)
    idx2user = {int(v): k for k, v in user2idx.items()}
    idx2item = {int(v): k for k, v in item2idx.items()}
    
    print(f"Loaded mappings - Users: {len(user2idx)}, Items: {len(item2idx)}")

    # Load original train data for user list
    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))
    train = train.sort_values(by=['user_session','event_time'])

    train['user_idx'] = train['user_id'].map(user2idx)
    train['item_idx'] = train['item_id'].map(item2idx)

    users = defaultdict(list)
    for u, i in zip(train['user_idx'], train['item_idx']):
        if pd.notna(u) and pd.notna(i):
            users[int(u)].append(int(i))

    # Popular items as fallback
    popular_top_10 = train.groupby('item_idx').size().sort_values(ascending=False)[:100].index.tolist()
    
    result = []
    
    print(f"Starting batch inference with GPU...")

    # ... (omitted) ...

    # Batch inference using RecBole's full_sort_topk
    user_list = list(users.keys())
    batch_size = 128  # Reduced from 256 for 8GB VRAM stability
    
    with torch.no_grad():
        for i in tqdm(range(0, len(user_list), batch_size)):
            batch_users = user_list[i:i+batch_size]
            
            for uid in batch_users:
                if str(uid) in dataset.field2token_id['user_idx']:
                    recbole_uid = dataset.token2id(dataset.uid_field, str(uid))
                    
                    try:
                        score, topk_iid = full_sort_topk([recbole_uid], model, test_data, k=args.topk, device=device)
                        predicted_items = dataset.id2token(dataset.iid_field, topk_iid.cpu())
                        predicted_item_list = [int(x) for x in predicted_items[0]]
                    except Exception as e:
                        predicted_item_list = popular_top_10[:args.topk]
                else:
                    predicted_item_list = popular_top_10[:args.topk]

                # [FIX] 올바른 역매핑 사용
                for iid in predicted_item_list:
                    if uid in idx2user and iid in idx2item:
                        result.append((idx2user[uid], idx2item[iid]))
                    elif uid in idx2user:
                        # iid가 없으면 인기 아이템으로 대체
                        for pop_iid in popular_top_10[:1]: # Fill with something valid if mapping fails
                            if pop_iid in idx2item:
                                result.append((idx2user[uid], idx2item[pop_iid]))
                                break

    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_path = os.path.join(args.output_dir, args.output_file)
    pd.DataFrame(result, columns=["user_id", "item_id"]).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    print(f"Total predictions: {len(result)}")
    
    # Verify output
    df = pd.read_csv(output_path)
    print(f"Unique users in output: {df['user_id'].nunique()}")
    print(f"Unique items in output: {df['item_id'].nunique()}")

if __name__ == "__main__":
    main()