# 커머스 추천 시스템 NDCG@10 개선 보고서

**문서 버전:** v1.0  
**작성일:** 2026-02-06  
**문서 유형:** 구현 계획서

---

## 1. 요약

본 보고서는 커머스 상품 구매 예측 태스크의 NDCG@10 지표 향상을 위한 ALS 베이스라인 모델 개선 방안을 제시합니다. 주요 개선 사항은 **이벤트 가중치 차등 부여**와 **시간 감쇠(Time Decay) 적용**입니다.

---

## 2. 현황 분석

### 2.1 현재 베이스라인의 문제점

현재 [`train_als.py`](file:///c:/Users/user/Downloads/Recsys/code/train_als.py) 코드는 모든 이벤트 타입에 대해 동일하게 `label=1`을 부여하고 있습니다.

**데이터셋 이벤트 분포:**
| 이벤트 타입 | 발생 건수 | 비율 |
|------------|-----------|------|
| view (조회) | 8,331,873 | 99.78% |
| cart (장바구니) | 16,362 | 0.20% |
| purchase (구매) | 2,076 | 0.02% |

**문제점:**
- 모든 사용자 행동을 동등하게 취급하여 **구매 의도의 강도를 반영하지 못함**
- 과거 데이터와 최근 데이터에 동일한 가중치를 부여하여 **현재 사용자 선호도를 정확히 반영하지 못함**

---

## 3. 개선 방안

### 3.1 이벤트 가중치 차등 부여

사용자의 구매 의도를 정확히 반영하기 위해 이벤트 타입별로 차등 가중치를 부여합니다.

**제안 가중치:**
```python
event_weight_mapping = {
    'view': 1.0,        # 기본 조회 행동
    'cart': 3.0,        # 장바구니 추가 (중간 관심도)
    'purchase': 5.0     # 실제 구매 (최고 관심도)
}
```

**근거:**
- **구매(purchase)**: 가장 강한 구매 의도를 나타내므로 최고 가중치(5.0) 부여
- **장바구니(cart)**: 구매 직전 단계로 중간 가중치(3.0) 부여
- **조회(view)**: 가장 낮은 관심도를 나타내므로 기본 가중치(1.0) 유지

### 3.2 시간 감쇠(Time Decay) 적용

최근 사용자 행동을 더 중요하게 고려하여 현재 선호도를 정확히 반영합니다.

**시간 감쇠 공식:**
```python
time_decay_weight = exp(-decay_rate × days_since_interaction)
```

**파라미터:**
- `decay_rate`: 감쇠율 (권장값: 0.01 ~ 0.05)
- `days_since_interaction`: 이벤트 발생일과 기준일(2020-02-29) 간의 일수

**효과:**
- 최근 이벤트는 높은 가중치 유지
- 오래된 이벤트는 지수적으로 가중치 감소
- 사용자의 현재 관심사를 더 정확히 반영

### 3.3 최종 가중치 계산

두 가지 가중치를 결합하여 최종 라벨을 계산합니다.

```python
final_label = event_weight × time_decay_weight
```

---

## 4. 구현 계획

### 4.1 수정 대상 파일

**[MODIFY]** [`train_als.py`](file:///c:/Users/user/Downloads/Recsys/code/train_als.py)

### 4.2 구현 단계

1. **이벤트 가중치 매핑 추가**
   - 46번 라인의 균일 라벨링(`label=1`) 제거
   - 이벤트 타입별 가중치 딕셔너리 적용

2. **시간 감쇠 함수 구현**
   - `event_time` 컬럼 파싱
   - 데이터셋 최신 날짜(2020-02-29)를 기준으로 일수 차이 계산
   - 지수 감쇠 공식 적용

3. **가중치 결합**
   - 이벤트 가중치와 시간 감쇠 가중치 곱셈
   - user-item 행렬 집계 전 적용

4. **모델 학습**
   - 기존 ALS 모델 하이퍼파라미터 유지
   - 가중치가 적용된 sparse matrix로 학습

### 4.3 코드 수정 위치

**기존 코드 (45-47번 라인):**
```python
# use the same confidence score for all event_types
train_df["label"] = 1
user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()
```

**개선 코드:**
```python
# Event weight mapping
event_weights = {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}
train_df['event_weight'] = train_df['event_type'].map(event_weights)

# Time decay
from datetime import datetime
import numpy as np
max_date = pd.to_datetime(train_df['event_time']).max()
train_df['days_diff'] = (max_date - pd.to_datetime(train_df['event_time'])).dt.days
decay_rate = 0.03
train_df['time_weight'] = np.exp(-decay_rate * train_df['days_diff'])

# Combined weight
train_df['label'] = train_df['event_weight'] * train_df['time_weight']
user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()
```

---

## 5. 검증 계획

### 5.1 자동화 테스트

1. **모델 학습 실행**
   ```bash
   python train_als.py
   ```

2. **출력 파일 생성 확인**
   - 경로: `../output/output.csv`
   - 형식: 샘플 제출 파일과 동일한지 확인

3. **데이터 무결성 검증**
   - 모든 사용자에 대해 10개 아이템 추천 여부 확인
   - user_id 및 item_id 매핑 정확성 확인

### 5.2 예상 효과

| 개선 사항 | 효과 |
|----------|------|
| **이벤트 가중치** | 구매/장바구니 이벤트가 추천에 더 큰 영향을 미쳐 사용자 의도 반영 향상 |
| **시간 감쇠** | 최근 브라우징 패턴 우선 반영으로 현재 관심사 정확도 향상 |
| **종합 효과** | NDCG@10 지표 개선 (구매 의도 계층 구조를 더 잘 포착) |

---

## 6. 결론

본 개선 방안은 기존 ALS 베이스라인의 단순한 균일 라벨링 방식을 탈피하여, **사용자 행동의 의미론적 중요도**와 **시간적 관련성**을 모두 반영합니다. 이를 통해 추천 시스템이 사용자의 실제 구매 의도를 더 정확히 예측할 수 있으며, NDCG@10 지표의 유의미한 향상이 기대됩니다.

**다음 단계:** 승인 후 구현 진행 → 학습 실행 → 결과 검증
