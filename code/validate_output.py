"""
앙상블 결과 검증 스크립트
- 출력 파일 형식 검증
- 사용자당 정확히 10개 아이템 확인
- 기본 통계 출력
"""

import pandas as pd
import argparse
import os

def validate_output(file_path):
    """앙상블 출력 파일 검증"""
    print("="*60)
    print(f"Validating: {file_path}")
    print("="*60)
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found - {file_path}")
        return False
    
    # 파일 로딩
    df = pd.read_csv(file_path)
    
    # 기본 정보
    print(f"\n[Basic Info]")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Unique items: {df['item_id'].nunique():,}")
    
    # 사용자당 아이템 수 확인
    user_item_counts = df.groupby('user_id').size()
    
    print(f"\n[Items per User]")
    print(f"  Min: {user_item_counts.min()}")
    print(f"  Max: {user_item_counts.max()}")
    print(f"  Mean: {user_item_counts.mean():.2f}")
    
    # 정확히 10개가 아닌 사용자 찾기
    invalid_users = user_item_counts[user_item_counts != 10]
    
    if len(invalid_users) > 0:
        print(f"\n⚠️  Warning: {len(invalid_users)} users have != 10 items")
        print(f"  Examples: {invalid_users.head().to_dict()}")
    else:
        print(f"\n✅ All users have exactly 10 items")
    
    # 중복 확인
    duplicates = df.duplicated(subset=['user_id', 'item_id']).sum()
    if duplicates > 0:
        print(f"\n⚠️  Warning: {duplicates} duplicate (user, item) pairs found")
    else:
        print(f"✅ No duplicates")
    
    # NULL 값 확인
    nulls = df.isnull().sum()
    if nulls.sum() > 0:
        print(f"\n⚠️  Warning: NULL values found")
        print(nulls[nulls > 0])
    else:
        print(f"✅ No NULL values")
    
    # 샘플 데이터 출력
    print(f"\n[Sample Data]")
    print(df.head(20))
    
    print("\n" + "="*60)
    
    if len(invalid_users) == 0 and duplicates == 0 and nulls.sum() == 0:
        print("✅ Validation PASSED - Ready for submission!")
    else:
        print("⚠️  Validation completed with warnings")
    
    print("="*60)
    
    return len(invalid_users) == 0

def compare_outputs(file1, file2):
    """두 출력 파일 비교"""
    print("\n" + "="*60)
    print("Comparing two output files")
    print("="*60)
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    print(f"\nFile 1: {file1}")
    print(f"  Users: {df1['user_id'].nunique():,}")
    print(f"  Items: {df1['item_id'].nunique():,}")
    
    print(f"\nFile 2: {file2}")
    print(f"  Users: {df2['user_id'].nunique():,}")
    print(f"  Items: {df2['item_id'].nunique():,}")
    
    # 공통 유저들의 추천 아이템 비교
    common_users = set(df1['user_id'].unique()) & set(df2['user_id'].unique())
    print(f"\nCommon users: {len(common_users):,}")
    
    if len(common_users) > 0:
        # 랜덤하게 5명 선택하여 비교
        import random
        sample_users = random.sample(list(common_users), min(5, len(common_users)))
        
        print(f"\n[Sample Comparison]")
        for user in sample_users:
            items1 = set(df1[df1['user_id'] == user]['item_id'])
            items2 = set(df2[df2['user_id'] == user]['item_id'])
            overlap = len(items1 & items2)
            print(f"  User {user}: {overlap}/10 items overlap")

def main():
    parser = argparse.ArgumentParser(description="Validate ensemble output")
    parser.add_argument("--file", type=str, required=True, help="Output file to validate")
    parser.add_argument("--compare_with", type=str, help="Second file to compare with")
    args = parser.parse_args()
    
    # 검증
    validate_output(args.file)
    
    # 비교 (선택)
    if args.compare_with and os.path.exists(args.compare_with):
        compare_outputs(args.file, args.compare_with)

if __name__ == "__main__":
    main()
