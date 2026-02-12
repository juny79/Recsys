"""
EASE 재학습 + L2 최적화
기존 EASE(l2=500)를 다양한 L2로 재학습하여 최적값 탐색
RTX 3070 8GB: Gram Matrix(26k x 26k)는 GPU에서 처리 가능
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
import argparse
import os
from tqdm import tqdm
import torch
import time

def train_ease(train_path, output_path, l2_reg, device, top_k=10):
    """EASE 모델 학습 및 추론"""
    
    print(f"\n{'='*60}")
    print(f"EASE Training (L2={l2_reg})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    print("Loading data...")
    df = pd.read_parquet(train_path)
    
    # Map IDs
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}
    
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {j: i for i, j in item2idx.items()}
    
    idx2user_arr = np.array([idx2user[i] for i in range(len(user_ids))])
    idx2item_arr = np.array([idx2item[i] for i in range(len(item_ids))])
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    print(f"Users: {n_users}, Items: {n_items}")
    
    # Create Sparse Matrix with Weights
    event_weights = {
        'purchase': 5.0,
        'cart': 3.0,
        'view': 1.0,
        'remove_from_cart': 0.0
    }
    df['weight'] = df['event_type'].map(event_weights).fillna(1.0)
    
    # Aggregate weights
    print("Aggregating interactions...")
    df_agg = df.groupby(['user_id', 'item_id'])['weight'].max().reset_index()
    
    rows = df_agg['user_id'].map(user2idx).values
    cols = df_agg['item_id'].map(item2idx).values
    data = df_agg['weight'].values
    
    X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    
    # EASE Training via PyTorch
    print("Calculating Gram Matrix (G = X^T X)...")
    G = X.transpose().dot(X).toarray()
    
    print(f"Moving G to {device}...")
    G_tensor = torch.from_numpy(G).float().to(device)
    
    # Add Regularization
    diag_indices = torch.arange(n_items).to(device)
    G_tensor[diag_indices, diag_indices] += l2_reg
    
    # Invert P = G^-1
    print("Inverting Gram Matrix (P = G^-1)...")
    P_tensor = torch.linalg.inv(G_tensor)
    
    # B = P / -diag(P)
    diag_P = P_tensor.diag()
    B_tensor = P_tensor / (-diag_P.view(-1, 1))
    B_tensor[diag_indices, diag_indices] = 0
    
    print("Generating Predictions...")
    
    # Inference Batching
    batch_size = 4096
    user_indices = np.arange(n_users)
    
    all_users = []
    all_items = []
    
    for start_idx in tqdm(range(0, n_users, batch_size), desc="Inference"):
        end_idx = min(start_idx + batch_size, n_users)
        batch_users = user_indices[start_idx:end_idx]
        
        X_batch_scipy = X[batch_users, :]
        
        # Convert to Sparse Tensor
        coo = X_batch_scipy.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(coo.data)
        shape = coo.shape
        
        X_batch_sparse = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)
        
        # Scores: Sparse @ Dense -> Dense
        S_batch_tensor = torch.sparse.mm(X_batch_sparse, B_tensor)
        
        # Set watched items to -inf (filter already liked)
        S_batch_tensor[i[0].to(device), i[1].to(device)] = -float('inf')
        
        # Top-K
        _, top_indices = torch.topk(S_batch_tensor, top_k, dim=1)
        
        top_indices_cpu = top_indices.cpu().numpy()
        batch_item_ids = idx2item_arr[top_indices_cpu]
        batch_user_ids_col = idx2user_arr[batch_users]
        batch_user_ids = np.repeat(batch_user_ids_col, top_k)
        
        all_users.append(batch_user_ids)
        all_items.append(batch_item_ids.flatten())
    
    final_users = np.concatenate(all_users)
    final_items = np.concatenate(all_items)
    
    df_out = pd.DataFrame({'user_id': final_users, 'item_id': final_items})
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df_out.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"✅ EASE (L2={l2_reg}) completed in {elapsed:.1f}s")
    print(f"   Predictions: {len(df_out):,}, Users: {df_out['user_id'].nunique():,}")
    print(f"   Saved to: {output_path}")
    
    return df_out

def main():
    parser = argparse.ArgumentParser(description="EASE Hyperparameter Tuning")
    parser.add_argument("--train_file", default="../data/train.parquet", type=str)
    parser.add_argument("--output_dir", default="../output/", type=str)
    parser.add_argument("--l2_reg", default=500.0, type=float, help="L2 Regularization (lambda)")
    parser.add_argument("--top_k", default=10, type=int, help="Top-K recommendations")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = os.path.join(args.output_dir, f"output_ease_l2_{int(args.l2_reg)}.csv")
    
    train_ease(args.train_file, output_path, args.l2_reg, device, args.top_k)

if __name__ == "__main__":
    main()
