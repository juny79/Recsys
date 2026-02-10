import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", required=True, type=str)
    parser.add_argument("--file2", required=True, type=str)
    args = parser.parse_args()
    
    print(f"Reading {args.file1}...")
    df1 = pd.read_csv(args.file1)
    print(f"Reading {args.file2}...")
    df2 = pd.read_csv(args.file2)
    
    # Calculate overlap per user
    # Assume 10 items per user
    
    # Group by user -> set of items
    print("Grouping...")
    u1 = df1.groupby('user_id')['item_id'].apply(set)
    u2 = df2.groupby('user_id')['item_id'].apply(set)
    
    # Align users
    common_users = u1.index.intersection(u2.index)
    print(f"Common Users: {len(common_users)}")
    
    u1 = u1[common_users]
    u2 = u2[common_users]
    
    # Calculate overlap
    print("Calculating Overlap...")
    overlaps = [len(s1.intersection(s2)) for s1, s2 in zip(u1, u2)]
    avg_overlap = sum(overlaps) / len(overlaps)
    
    print(f"Average Overlap (Intersection Size @ 10): {avg_overlap:.4f}")
    print(f"Jaccard Index: {avg_overlap / (20 - avg_overlap):.4f}") # Approx Jaccard if set size is fixed at 10

if __name__ == "__main__":
    main()
