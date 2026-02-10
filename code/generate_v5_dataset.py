import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--als_output", default="output/output_decay01_14.csv", type=str)
    parser.add_argument("--sasrec_output", default="output/output_sasrec_top100.csv", type=str)
    parser.add_argument("--global_pop", default="data/global_popularity.parquet", type=str)
    parser.add_argument("--train_file", default="data/train.parquet", type=str)
    parser.add_argument("--output_dir", default="data/", type=str)
    args = parser.parse_args()

    # 1. Load Shared Resources
    print("Loading Global Popularity...")
    pop = pd.read_parquet(args.global_pop)
    pop_top100 = pop.sort_values('pop_count', ascending=False).head(100)
    pop_top100['pop_rank'] = range(1, 101)
    del pop
    
    print("Loading Item Stats...")
    item_stats = pd.read_parquet("data/item_stats_v3.parquet")
    
    print("Loading User Trajectories...")
    user_traj = pd.read_parquet("data/user_trajectory_v3.parquet")
    
    print("Loading User Last State & History (Leakage Fix)...")
    df_train = pd.read_parquet(args.train_file)
    df_train = df_train.sort_values(['user_id', 'event_time'])
    
    # [FIX] Time-split for Leakage Prevention
    # df_last corresponds to the TRUE target for training
    df_last = df_train.groupby('user_id').tail(1).copy()
    # df_past is used to generate features for the training target
    df_past = df_train.drop(df_last.index)
    
    # 1.1 Features for Training (using df_past)
    user_last_state_train = df_past.groupby('user_id').tail(1).copy()
    user_last_state_train['last_hour'] = pd.to_datetime(user_last_state_train['event_time']).dt.hour
    user_last_state_train['last_day'] = pd.to_datetime(user_last_state_train['event_time']).dt.dayofweek
    
    label_train = df_last[['user_id', 'item_id']].copy()
    label_train['label'] = 1
    
    print("Computing history sets for training...")
    history_dict_train = df_past.groupby('user_id')['item_id'].apply(lambda x: set(list(x)[-50:])).to_dict()
    
    # 1.2 Features for Test/Inference (using full df_train)
    user_last_state_test = df_train.groupby('user_id').tail(1).copy()
    user_last_state_test['last_hour'] = pd.to_datetime(user_last_state_test['event_time']).dt.hour
    user_last_state_test['last_day'] = pd.to_datetime(user_last_state_test['event_time']).dt.dayofweek
    
    print("Computing history sets for test...")
    history_dict_test = df_train.groupby('user_id')['item_id'].apply(lambda x: set(list(x)[-50:])).to_dict()
    
    del df_train, df_past, df_last
    gc.collect()

    # 2. Load Source Candidates
    print("Loading ALS & SASRec Candidates...")
    als = pd.read_csv(args.als_output)
    als['als_rank'] = als.groupby('user_id').cumcount() + 1
    
    sas = pd.read_csv(args.sasrec_output)
    sas['sas_rank'] = sas.groupby('user_id').cumcount() + 1
    
    all_users = pd.concat([als['user_id'], sas['user_id']]).unique()
    print(f"Total Unique Users: {len(all_users)}")

    # 3. Chunked Processing
    print(f"Processing users in chunks...")
    chunk_size = 50000
    train_parts = []
    
    # We'll save test candidates directly to a parquet file in chunks if possible, 
    # but for simplicity and to ensure Top-100 logic, we'll collect then save.
    test_candidates_parts = []
    
    def check_membership_fast(user_id, item_id, h_dict):
        history = h_dict.get(user_id)
        if history:
            return 1 if item_id in history else 0
        return 0
    v_check = np.vectorize(check_membership_fast, excluded=['h_dict'])

    for i in tqdm(range(0, len(all_users), chunk_size)):
        user_chunk = all_users[i:i+chunk_size]
        
        # Build Union for this chunk
        als_c = als[als['user_id'].isin(user_chunk)]
        sas_c = sas[sas['user_id'].isin(user_chunk)]
        
        pop_c = pd.DataFrame({'user_id': np.repeat(user_chunk, 100)})
        pop_c['item_id'] = np.tile(pop_top100['item_id'].values, len(user_chunk))
        pop_c = pd.merge(pop_c, pop_top100[['item_id', 'pop_rank', 'pop_count']], on='item_id', how='left')
        
        candidates = pd.merge(
            als_c[['user_id', 'item_id', 'als_rank']],
            sas_c[['user_id', 'item_id', 'sas_rank']],
            on=['user_id', 'item_id'],
            how='outer'
        )
        # candidates now has Union of ALS + SASRec
        candidates = pd.merge(candidates, pop_c, on=['user_id', 'item_id'], how='outer')
        
        # [FIX] Heuristics Re-balancing (More weight to Models)
        K_val = 20
        candidates['als_score'] = 1 / (K_val + candidates['als_rank'].fillna(500))
        candidates['sas_score'] = 1 / (K_val + candidates['sas_rank'].fillna(500))
        candidates['pop_score'] = np.log1p(candidates['pop_count'].fillna(0))
        
        # v5_score: prioritize models over pure popularity
        candidates['v5_score'] = (candidates['als_score'] * 0.45 + 
                                  candidates['sas_score'] * 0.45 + 
                                  (candidates['pop_score'] / 20) * 0.1)
        
        # 1. Processing for TRAINING (Leakage-free)
        chunk_train = candidates.copy()
        chunk_train = pd.merge(chunk_train, user_last_state_train[['user_id', 'last_hour', 'last_day']], on='user_id', how='left')
        chunk_train = pd.merge(chunk_train, user_traj, on='user_id', how='left')
        chunk_train = pd.merge(chunk_train, item_stats, on='item_id', how='left')
        
        # is_repeat for training (MUST use history_dict_train)
        chunk_train['is_repeat'] = v_check(chunk_train['user_id'], chunk_train['item_id'], history_dict_train)
        chunk_train['brand_match'] = (chunk_train['item_brand'] == chunk_train['top_brand']).astype(int)
        
        # Labeling
        chunk_train = pd.merge(chunk_train, label_train, on=['user_id', 'item_id'], how='left')
        chunk_train['label'] = chunk_train['label'].fillna(0)
        
        # Negative Sampling
        pos = chunk_train[chunk_train['label'] == 1]
        neg = chunk_train[chunk_train['label'] == 0]
        if len(pos) > 0:
            neg_sample = neg.sample(n=min(len(neg), len(pos)*10), random_state=42)
            train_parts.append(pd.concat([pos, neg_sample]))
        
        # 2. Processing for TEST/INFERENCE (No leakage concern here)
        chunk_test = candidates.copy()
        chunk_test = pd.merge(chunk_test, user_last_state_test[['user_id', 'last_hour', 'last_day']], on='user_id', how='left')
        chunk_test = pd.merge(chunk_test, user_traj, on='user_id', how='left')
        chunk_test = pd.merge(chunk_test, item_stats, on='item_id', how='left')
        
        # is_repeat for test (CAN use history_dict_test)
        chunk_test['is_repeat'] = v_check(chunk_test['user_id'], chunk_test['item_id'], history_dict_test)
        chunk_test['brand_match'] = (chunk_test['item_brand'] == chunk_test['top_brand']).astype(int)
        
        # Keep only Top 100 for inference
        top100_c = chunk_test.sort_values(['user_id', 'v5_score'], ascending=[True, False]).groupby('user_id').head(100)
        test_candidates_parts.append(top100_c)
        
        del als_c, sas_c, pop_c, candidates, chunk_train, chunk_test, top100_c
        gc.collect()

    # 4. Final Save
    print("Combining & Saving Final Datasets...")
    train_final = pd.concat(train_parts).sample(frac=1.0, random_state=42)
    train_final.to_parquet(os.path.join(args.output_dir, "ltr_v5_train.parquet"))
    del train_final, train_parts
    gc.collect()
    
    print("Saving test candidates...")
    test_final = pd.concat(test_candidates_parts)
    # Important to handle NaNs in categorical columns for CatBoost
    test_final.to_parquet(os.path.join(args.output_dir, "ltr_v5_test_candidates.parquet"), row_group_size=100000)
    
    print("Success: V5 Dataset Generated.")

if __name__ == "__main__":
    main()
