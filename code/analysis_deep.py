"""데이터 심층 분석: nDCG 극대화를 위한 새로운 시그널 탐색"""
import pandas as pd
import numpy as np

df = pd.read_parquet('../data/train.parquet')
df['event_time'] = pd.to_datetime(df['event_time'])

print('=== 기본 통계 ===')
print(f'전체 상호작용: {len(df):,}')
print(f'유저 수: {df.user_id.nunique():,}')
print(f'아이템 수: {df.item_id.nunique():,}')
print(f'\n컬럼: {df.columns.tolist()}')
print(f'\n샘플:\n{df.head(3)}')

# 유저별 활동량 분포
user_counts = df.groupby('user_id').size()
print(f'\n=== 유저 활동량 분포 ===')
bins = [0, 1, 5, 20, 50, 100, float('inf')]
labels = ['1건', '2-5건', '6-20건', '21-50건', '51-100건', '100+건']
segs = pd.cut(user_counts, bins=bins, labels=labels)
for label in labels:
    cnt = (segs == label).sum()
    pct = cnt / len(user_counts) * 100
    print(f'  {label}: {cnt:,} ({pct:.1f}%)')
print(f'  중앙값: {user_counts.median():.0f}, 평균: {user_counts.mean():.1f}')

# 이벤트 타입 분포
print(f'\n=== 이벤트 타입 분포 ===')
for et, cnt in df['event_type'].value_counts().items():
    print(f'  {et}: {cnt:,} ({cnt/len(df)*100:.1f}%)')

purchase_users = df[df.event_type == 'purchase'].user_id.nunique()
cart_users = df[df.event_type == 'cart'].user_id.nunique()
print(f'\n구매 유저: {purchase_users:,} ({purchase_users/df.user_id.nunique()*100:.1f}%)')
print(f'장바구니 유저: {cart_users:,} ({cart_users/df.user_id.nunique()*100:.1f}%)')

# 시간 범위
print(f'\n=== 시간 범위 ===')
print(f'시작: {df.event_time.min()}')
print(f'종료: {df.event_time.max()}')

max_time = df.event_time.max()

# 최근 기간별 통계
for days in [1, 3, 7, 14, 30]:
    recent = df[df.event_time >= max_time - pd.Timedelta(days=days)]
    print(f'최근 {days:2d}일: {len(recent):>10,} ({len(recent)/len(df)*100:5.1f}%), 유저:{recent.user_id.nunique():>8,}, 아이템:{recent.item_id.nunique():>7,}')

# 공동구매 패턴
print(f'\n=== 공동구매 패턴 ===')
purchases = df[df.event_type == 'purchase']
user_purch_items = purchases.groupby('user_id')['item_id'].nunique()
print(f'구매 유저: {len(user_purch_items):,}')
print(f'2+ 아이템: {(user_purch_items >= 2).sum():,} ({(user_purch_items >= 2).mean()*100:.1f}%)')
print(f'3+ 아이템: {(user_purch_items >= 3).sum():,} ({(user_purch_items >= 3).mean()*100:.1f}%)')

# 아이템 인기도 (최근 vs 전체)
print(f'\n=== 아이템 인기도 트렌드 ===')
last_7d = df[df.event_time >= max_time - pd.Timedelta(days=7)]
item_pop_all = df.groupby('item_id').size()
item_pop_7d = last_7d.groupby('item_id').size()
# 전체 상위 100 아이템 중 최근 7일에도 활발한 비율
top100_all = item_pop_all.nlargest(100).index
top100_7d = item_pop_7d.nlargest(100).index
overlap = len(set(top100_all) & set(top100_7d))
print(f'Top-100 전체 vs 최근7일 겹침: {overlap}/100 ({overlap}%)')

# 카테고리 정보 확인
if 'category_id' in df.columns:
    print(f'\n=== 카테고리 정보 ===')
    print(f'카테고리 수: {df.category_id.nunique():,}')
    cat_per_item = df.groupby('item_id')['category_id'].nunique()
    print(f'아이템당 카테고리 수 (중앙값): {cat_per_item.median():.0f}')

# 유저별 마지막 행동 분석
print(f'\n=== 유저 마지막 행동 분석 ===')
last_actions = df.sort_values('event_time').groupby('user_id').last()
print(last_actions['event_type'].value_counts())

# 재구매 패턴 상세
print(f'\n=== 재구매 패턴 상세 ===')
user_item_counts = df.groupby(['user_id', 'item_id']).size().reset_index(name='cnt')
repeat = user_item_counts[user_item_counts.cnt >= 2]
print(f'재방문 (user, item) 쌍: {len(repeat):,} ({len(repeat)/len(user_item_counts)*100:.1f}%)')
print(f'3회+ 방문: {(user_item_counts.cnt >= 3).sum():,} ({(user_item_counts.cnt >= 3).mean()*100:.1f}%)')

# 재방문 아이템의 구매 전환율
repeat_users_items = set(zip(repeat.user_id, repeat.item_id))
purchase_pairs = set(zip(purchases.user_id, purchases.item_id))
repeat_purchased = len(repeat_users_items & purchase_pairs)
print(f'재방문 중 구매: {repeat_purchased:,} ({repeat_purchased/len(repeat_users_items)*100:.1f}%)')
