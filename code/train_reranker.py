import pandas as pd
import numpy as np
import xgboost as xgb
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data", type=str)
    parser.add_argument("--output_path", default="../output/output_reranked.csv", type=str)
    args = parser.parse_args()
    
    # 1. Load Data
    print("Loading Datasets...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train_ranker.parquet"))
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test_ranker_candidates.parquet"))
    
    user_feat = pd.read_parquet(os.path.join(args.data_dir, "user_features.parquet"))
    item_feat = pd.read_parquet(os.path.join(args.data_dir, "item_features.parquet"))
    
    # 2. Merge Features
    print("Merging Features for Train...")
    train_df = pd.merge(train_df, user_feat, on='user_id', how='left')
    train_df = pd.merge(train_df, item_feat, on='item_id', how='left')
    
    print("Merging Features for Test...")
    test_df = pd.merge(test_df, user_feat, on='user_id', how='left')
    test_df = pd.merge(test_df, item_feat, on='item_id', how='left')
    
    # Drop IDs and Timestamp for training
    # Keep useful columns
    feature_cols = [
        'user_activity', 'user_avg_price', 
        'price_mean', 'price_min', 'price_max', 'pop_count', 'brand', 'category_code',
        'view', 'cart', 'purchase', 'conversion_rate'
    ]
    # 'last_event_time' might be useful as diff
    
    # Fill NA
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(0)
        test_df[col] = test_df[col].fillna(0)
        
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    X_test = test_df[feature_cols]
    
    # 3. Train XGBoost
    print("Training XGBoost...")
    # Use GPU if available
    # tree_method='gpu_hist' works on linux/windows if cuda is available
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        tree_method='hist', # 'gpu_hist' might fail if not compiled with gpu
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Feature Importance
    print("Feature Importance:")
    print(pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False))
    
    # 4. Predict
    print("Predicting...")
    # We predict probability of class 1
    test_df['score'] = model.predict_proba(X_test)[:, 1]
    
    # 5. Sort and Select Top 10
    print("Sorting and Selecting Top 10...")
    final_df = test_df.sort_values(by=['user_id', 'score'], ascending=[True, False])
    final_top10 = final_df.groupby('user_id').head(10)
    
    # Save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
        
    final_top10[['user_id', 'item_id']].to_csv(args.output_path, index=False)
    print(f"Reranking completed. Saved to {args.output_path}")
    print(f"Total predictions: {len(final_top10)}")

if __name__ == "__main__":
    main()
