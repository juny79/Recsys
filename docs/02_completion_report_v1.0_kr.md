# 커머스 추천 시스템 NDCG@10 개선 - 작업 완료 보고서

**문서 버전:** v1.0  
**작성일:** 2026-02-06 01:09 KST  
**문서 유형:** 구현 완료 보고서

---

## 📋 작업 요약

ALS 베이스라인 모델에 **이벤트 가중치**와 **시간 감쇠** 기능을 추가하여 NDCG@10 성능 향상을 위한 개선을 완료했습니다.

---

## ✅ 완료된 작업

### 1. 이벤트 가중치 차등 부여

**구현 위치:** [`train_als.py`](file:///c:/Users/user/Downloads/Recsys/code/train_als.py#L45-L47)

```python
# Event weight mapping: differentiate by purchase intent
event_weights = {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}
train_df['event_weight'] = train_df['event_type'].map(event_weights)
```

**효과:**
- `view`: 기본 가중치 1.0
- `cart`: 3배 가중치 (장바구니 추가는 구매 의도 중간 단계)
- `purchase`: 5배 가중치 (실제 구매는 최고 관심도)

### 2. 시간 감쇠(Time Decay) 적용

**구현 위치:** [`train_als.py`](file:///c:/Users/user/Downloads/Recsys/code/train_als.py#L49-L53)

```python
# Time decay: prioritize recent interactions
max_date = pd.to_datetime(train_df['event_time']).max()
train_df['days_diff'] = (max_date - pd.to_datetime(train_df['event_time'])).dt.days
decay_rate = 0.03  # Configurable decay rate
train_df['time_weight'] = np.exp(-decay_rate * train_df['days_diff'])
```

**효과:**
- 최근 이벤트에 높은 가중치 부여
- 과거 이벤트는 지수적으로 감소
- `decay_rate=0.03` 설정으로 약 33일마다 가중치가 절반으로 감소

### 3. 최종 가중치 결합

**구현 위치:** [`train_als.py`](file:///c:/Users/user/Downloads/Recsys/code/train_als.py#L55-L56)

```python
# Combined weight: event_weight × time_weight
train_df['label'] = train_df['event_weight'] * train_df['time_weight']
user_item_matrix = train_df.groupby(["user_idx", "item_idx"])["label"].sum().reset_index()
```

---

## 🔧 GPU 활용 시도

**시도 내역:**
- 초기에 `use_gpu=True` 설정하여 RTX 3070 활용 시도
- `implicit` 라이브러리의 CUDA 확장이 빌드되지 않은 버전 설치됨
- CPU 모드(`use_gpu=False`)로 전환하여 정상 실행

> [!NOTE]
> GPU 가속이 필요한 경우, CUDA 지원 버전의 `implicit` 라이브러리를 별도로 컴파일해야 합니다.

---

## 📊 검증 결과

### 모델 학습 완료

```
Training Status: ✅ Success
Exit Code: 0
```

### 출력 파일 생성

**파일 경로:** [`../output/output.csv`](file:///c:/Users/user/Downloads/Recsys/output/output.csv)

**파일 정보:**
- 총 크기: 478.7 MB
- 총 추천 건수: 6,382,570개
- 고유 사용자 수: 638,257명
- 사용자당 추천 아이템: 10개

**데이터 형식:**
| user_id | item_id |
|---------|---------|
| cd397d... | 583265... |
| cd397d... | 1ea95b... |
| ... | ... |

---

## 📈 주요 개선 사항 요약

| 개선 항목 | 기존 방식 | 개선 방식 | 기대 효과 |
|----------|----------|----------|----------|
| **이벤트 처리** | 모든 이벤트 동일 (`label=1`) | 차등 가중치 (view=1, cart=3, purchase=5) | 구매 의도가 높은 행동 우선 반영 |
| **시간 처리** | 모든 시점 동일 | 지수 감쇠 (`exp(-0.03 × days)`) | 최근 관심사 정확히 반영 |
| **추천 품질** | 단순 빈도 기반 | 의도+시간 복합 반영 | NDCG@10 향상 |

---

## 🎯 다음 단계

1. **리더보드 제출**
   - 파일: [`output.csv`](file:///c:/Users/user/Downloads/Recsys/output/output.csv)
   - 형식: 샘플 제출 파일과 동일

2. **하이퍼파라미터 튜닝 (선택사항)**
   - `decay_rate`: 현재 0.03 → 0.01~0.05 범위 실험
   - 이벤트 가중치: view=1, cart=3, purchase=5 → 다른 비율 실험
   - ALS 파라미터: `num_factor`, `regularization`, `alpha` 조정

3. **추가 개선 방안**
   - 카테고리 정보 활용
   - 가격 정보 활용
   - 사용자 세션 정보 활용

---

## 📁 수정된 파일

render_diffs(file:///c:/Users/user/Downloads/Recsys/code/train_als.py)

---

**실행 환경:** Windows, CPU Mode (RTX 3070 available)  
**생성된 제출 파일:** `c:\Users\user\Downloads\Recsys\output\output.csv`
