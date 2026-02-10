import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import os 
from utils import *
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--train_dataset", default="train.parquet", type=str)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    train = pd.read_parquet(os.path.join(args.data_dir, args.train_dataset))
    train['event_time'] = pd.to_datetime(train['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
    train = train.sort_values(by=['user_session','event_time'])

    # [FIX] 시퀀스 복제 제거 - SASRec은 순차 패턴을 학습하므로 원본 시퀀스 유지
    # 이벤트 가중치는 복제가 아닌 다른 방식으로 적용해야 함
    print(f"Total interactions: {len(train)}")
    
    # 원본 데이터 그대로 사용 (시퀀스 보존)
    train_df = train[['user_id','item_id','user_session','event_time']].copy()
    train_df['event_time'] = train_df['event_time'].values.astype(float)

    # ID 매핑 생성 (원본 데이터 기준)
    user2idx = {v: k for k, v in enumerate(train_df['user_id'].unique())}
    item2idx = {v: k for k, v in enumerate(train_df['item_id'].unique())}

    print(f"Unique users: {len(user2idx)}")
    print(f"Unique items: {len(item2idx)}")

    # [FIX] 역매핑도 함께 저장 (추론 시 일관성 보장)
    idx2user = {v: k for k, v in user2idx.items()}
    idx2item = {v: k for k, v in item2idx.items()}

    with open(os.path.join(args.data_dir,'user2idx.json'),"w") as f_user:
        json.dump(user2idx, f_user)

    with open(os.path.join(args.data_dir,'item2idx.json'),"w") as f_item:
        json.dump(item2idx, f_item)

    # [NEW] 역매핑 파일도 저장
    with open(os.path.join(args.data_dir,'idx2user.json'),"w") as f:
        json.dump(idx2user, f)

    with open(os.path.join(args.data_dir,'idx2item.json'),"w") as f:
        json.dump(idx2item, f)

    # Apply the mapping functions to 'user_id' and 'item_id' columns
    train_df['user_idx'] = train_df['user_id'].map(user2idx)
    train_df['item_idx'] = train_df['item_id'].map(item2idx)

    train_df = train_df.dropna().reset_index(drop=True)
    train_df.rename(columns={'user_idx': 'user_idx:token', 'item_idx': 'item_idx:token', 'event_time': 'event_time:float'}, inplace=True)
    
    outdir = os.path.join(args.data_dir,'SASRec_dataset')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_df[['user_idx:token', 'item_idx:token', 'event_time:float']].to_csv(os.path.join(outdir,'SASRec_dataset.inter'), sep='\t', index=None)
    print('Recbole dataset generated (original sequence preserved)')

if __name__ == "__main__":
    main()
