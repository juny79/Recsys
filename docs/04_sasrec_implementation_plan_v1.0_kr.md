# SASRec 모델 개선 구현 계획

**문서 버전:** v1.0  
**작성일:** 2026-02-06  
**문서 유형:** 구현 계획서  
**목표:** NDCG@10 점수 극대화 (현재 ALS: 0.0955)

---

## 1. 현황 분석

### 1.1 현재 SASRec 베이스라인

**모델 설정 ([`sasrec.yaml`](file:///c:/Users/user/Downloads/Recsys/code/yaml/sasrec.yaml)):**
```yaml
n_layers: 2           # Transformer 레이어 수
n_heads: 4            # Attention 헤드 수
inner_size: 256       # Feed-forward 크기
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.3
MAX_ITEM_LIST_LENGTH: 50
loss_type: 'BPR'      # Bayesian Personalized Ranking
epochs: 20
train_batch_size: 4096
```

**데이터 준비 ([`recbole_dataset.py`](file:///c:/Users/user/Downloads/Recsys/code/recbole_dataset.py)):**
- ❌ 이벤트 가중치 미적용
- ❌ 시간 감쇠 미적용
- ❌ event_type 정보 미사용
- ✅ 시간순 정렬만 수행

**문제점:**
1. **모델 용량 부족**: n_layers=2, n_heads=4는 830만 건 데이터에 비해 가벼움
2. **단순 Negative Sampling**: 랜덤 아이템 선택
3. **이벤트 타입 무시**: view/cart/purchase 구분 없음
4. **시간 감쇠 미적용**: 과거 데이터와 최근 데이터 동일 취급

---

## 2. 개선 방안

### 2.1 모델 용량 증가 (Hyperparameter Tuning)

**목표:** Transformer 모델의 표현력 향상

**변경 사항:**
```yaml
# Before
n_layers: 2
n_heads: 4
inner_size: 256

# After
n_layers: 3          # +50% 증가
n_heads: 8           # +100% 증가
inner_size: 512      # +100% 증가
```

**근거:**
- 데이터 크기: 830만 건 → 모델 용량 증가 정당화
- RecSys 논문 기준: n_layers=2-4, n_heads=4-8이 일반적
- 계산 비용 증가하지만 RTX 3070으로 처리 가능

### 2.2 Hard Negative Sampling

**현재 방식:**
- RecBole의 기본 BPR loss: 랜덤하게 negative 샘플 선택

**개선 방식:**
- **Hard Negative**: "조회했지만 구매하지 않은 아이템"
- 사용자가 관심은 있었지만 최종 선택하지 않은 아이템을 negative로 사용

**구현 방법:**
1. `recbole_dataset.py`에서 데이터 준비 시:
   - view만 있고 purchase 없는 아이템 식별
   - 이를 hard negative 후보로 마킹

2. 커스텀 Sampler 또는 데이터 증강 적용

> [!NOTE]
> RecBole에서 커스텀 negative sampling은 복잡하므로, 우선 이벤트 가중치로 간접 구현 가능

### 2.3 이벤트 가중치 추가

**ALS와 동일한 가중치 적용:**

```python
event_weights = {
    'view': 1.0,
    'cart': 3.0,
    'purchase': 5.0
}
```

**구현 위치:** [`recbole_dataset.py`](file:///c:/Users/user/Downloads/Recsys/code/recbole_dataset.py)

**방법:**
- 각 interaction에 가중치 부여
- purchase 이벤트는 여러 번 복제하여 중요도 반영
- 또는 RecBole의 interaction weight 기능 활용

### 2.4 시간 감쇠 적용

**ALS와 동일한 공식:**
```python
max_date = train['event_time'].max()
train['days_diff'] = (max_date - train['event_time']).dt.days
decay_rate = 0.03
train['time_weight'] = np.exp(-decay_rate * train['days_diff'])
```

**활용 방법:**
- 시간 가중치를 이벤트 가중치와 곱하여 최종 중요도 결정
- 최근 interaction일수록 높은 빈도로 학습 데이터에 포함

---

## 3. 구현 계획

### 3.1 수정 대상 파일

1. **[MODIFY]** [`yaml/sasrec.yaml`](file:///c:/Users/user/Downloads/Recsys/code/yaml/sasrec.yaml)
   - 하이퍼파라미터 증가

2. **[MODIFY]** [`recbole_dataset.py`](file:///c:/Users/user/Downloads/Recsys/code/recbole_dataset.py)
   - 이벤트 가중치 추가
   - 시간 감쇠 추가
   - event_type 컬럼 포함

3. **[NO CHANGE]** [`train_sasrec.py`](file:///c:/Users/user/Downloads/Recsys/code/train_sasrec.py)
   - 기존 학습 코드 그대로 사용

4. **[NO CHANGE]** [`inference_sasrec.py`](file:///c:/Users/user/Downloads/Recsys/code/inference_sasrec.py)
   - 기존 추론 코드 그대로 사용

### 3.2 구현 단계

#### Step 1: 하이퍼파라미터 최적화
```yaml
# sasrec.yaml 수정
n_layers: 3
n_heads: 8
inner_size: 512
hidden_dropout_prob: 0.4  # 과적합 방지 위해 약간 감소
attn_dropout_prob: 0.2
epochs: 30  # 더 많은 epoch로 수렴 보장
```

#### Step 2: 데이터셋 준비 개선
```python
# recbole_dataset.py 수정

# 원본 데이터에서 event_type 유지
train_df = train[['user_id','item_id','user_session','event_time','event_type']]

# 이벤트 가중치 적용
event_weights = {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}
train_df['event_weight'] = train_df['event_type'].map(event_weights)

# 시간 감쇠 적용
max_date = train_df['event_time'].max()
train_df['days_diff'] = (max_date - train_df['event_time']).dt.days
decay_rate = 0.03
train_df['time_weight'] = np.exp(-decay_rate * train_df['days_diff'])

# 최종 가중치
train_df['final_weight'] = train_df['event_weight'] * train_df['time_weight']

# 가중치에 따라 interaction 복제 (정수로 반올림)
# 또는 RecBole이 지원하는 경우 weight 컬럼으로 전달
```

#### Step 3: Hard Negative Sampling (간접 구현)

view만 있고 purchase가 없는 패턴을 학습 시 더 강조:
- purchase 이벤트를 5배 복제
- view 이벤트는 원래대로
- 이를 통해 모델이 purchase로 이어질 패턴 강화 학습

---

## 4. 실행 계획

### 4.1 실행 순서

```bash
# 1. 개선된 데이터셋 준비
python recbole_dataset.py

# 2. SASRec 모델 학습
python train_sasrec.py

# 3. 추론 및 결과 생성
python inference_sasrec.py --model_file ./saved/SASRec.pth
```

### 4.2 예상 학습 시간

- **데이터셋 준비**: 2-3분
- **모델 학습**: 30 epochs × 약 5-10분/epoch = **2.5-5시간**
- **추론**: 10-15분

> [!WARNING]
> 모델 용량 증가로 학습 시간이 베이스라인 대비 2-3배 증가할 수 있습니다.

---

## 5. 기대 효과

### 5.1 예상 NDCG@10 향상

| 개선 사항 | 예상 기여도 | 누적 점수 |
|----------|-----------|----------|
| **베이스라인** | - | 0.08-0.10 |
| + 이벤트 가중치 | +10-15% | 0.09-0.11 |
| + 시간 감쇠 | +5-10% | 0.10-0.12 |
| + 하이퍼파라미터 | +15-25% | **0.12-0.15** |

**최종 목표:** NDCG@10 ≥ 0.12 (ALS 0.0955 대비 25% 향상)

### 5.2 SASRec vs ALS 비교

| 특징 | ALS | SASRec |
|------|-----|--------|
| **모델 타입** | 행렬 분해 | Transformer |
| **순차 패턴** | ❌ | ✅ |
| **계산 비용** | 낮음 | 높음 |
| **예상 점수** | 0.0955 | 0.12-0.15 |

---

## 6. 검증 계획

### 6.1 학습 모니터링

- Validation NDCG@10 추적
- Early stopping으로 과적합 방지
- 학습 곡선 확인

### 6.2 결과 비교

1. **ALS vs SASRec 점수 비교**
2. **시간-성능 trade-off 분석**
3. **앙상블 가능성 검토**

---

## 7. 다음 단계

### 우선순위 1: 기본 개선 (필수)
- [x] 계획 수립
- [ ] 하이퍼파라미터 튜닝
- [ ] 이벤트 가중치 & 시간 감쇠 추가
- [ ] 학습 및 검증

### 우선순위 2: 고급 개선 (선택)
- [ ] 진짜 Hard Negative Sampling 구현
- [ ] 앙상블 (ALS + SASRec)
- [ ] 추가 피처 (카테고리, 가격, 브랜드)

---

**작성일:** 2026-02-06  
**다음 작업:** 구현 시작
