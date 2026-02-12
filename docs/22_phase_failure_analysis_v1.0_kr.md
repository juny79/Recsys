# Phase 2-3 실패 원인 분석 및 대응 전략

## 📊 성능 추이

| Phase | nDCG@10 | 변화 | 전략 |
|-------|---------|------|------|
| Baseline | 0.1173 | - | 기본 앙상블 |
| Fast | 0.1265 | +7.8% | 기본 Repeat Boost |
| **Phase 1** | **0.1330** | **+5.1%** | **Event Type + Recency (성공)** |
| Phase 2 | 0.1331 | +0.08% | Category + Diversity (거의 효과 없음) |
| Phase 3 | 0.1322 | -0.68% | **강한 Boost (실패)** |

---

## 🔍 Phase 2 실패 원인

### 실패한 전략
1. **Category Consistency Boost** (1.3~1.6배)
2. **Diversity Penalty** (4-10위 다양성)

### 왜 실패했나?
- **Category 정보의 한계**: 68.4% 일관성이지만, 이미 ALS/SASRec이 학습함
- **Diversity는 nDCG에 불리**: nDCG는 정확도 중심 지표
- **추가 효과 미미**: Category는 이미 모델에 반영됨

---

## 🔍 Phase 3 실패 원인 (더 심각)

### 적용한 변경사항
```python
# 1. Repeat Boost 강화
count >= 3: 2.5 → 2.8
count == 2: 2.0 → 2.2

# 2. Event Boost 강화  
purchase: 1.8 → 2.2
cart: 1.4 → 1.6

# 3. Purchase Priority 추가
has_purchase: +100 bonus points

# 4. Position-aware Boost
rank 1: 2.0배
rank 2: 1.8배
rank 3: 1.6배
```

### 왜 실패했나? (핵심 원인)

#### 1. **오버부스팅 (Over-boosting)**
- Purchase priority +100은 너무 극단적
- 다른 좋은 후보들이 완전히 밀려남
- 예: Purchase 있는 평범한 아이템 > Purchase 없는 최고 후보

#### 2. **Position-aware의 역효과**
- Top-3를 너무 고정시킴
- 원래 2-3위였던 더 나은 아이템이 4-5위로 밀림
- nDCG는 위치 민감 → 좋은 아이템이 한 칸만 밀려도 큰 손실

#### 3. **Boost 값의 곱셈 효과**
```python
# Phase 3의 극단적 예시
repeat(2.8) × event(2.2) × recency(2.0) × position(2.0) 
= 24.64배!

# 반면 Phase 1 (성공)
repeat(2.5) × event(1.8) × recency(2.0) 
= 9배 (적당함)
```

#### 4. **데이터의 본질 무시**
- Purchase는 희귀 (전체의 ~5%)
- 너무 강조하면 다양성 손실
- 실제로는 view → cart → purchase 순서가 자연스러움

---

## ✅ 학습한 교훈

### 1. **Phase 1이 Sweet Spot**
- Event Type 차등: 1.8배 (purchase), 1.4배 (cart) ✅
- Repeat: 2.5배 (3회 이상) ✅
- Recency: 2.0배 (1시간 이내) ✅
- **이 조합이 최적**

### 2. **더 강하다고 더 좋은 것 아님**
- Boost는 "적당히"가 중요
- 곱셈 효과를 고려해야 함
- 극단적 값은 역효과

### 3. **Category는 이미 학습됨**
- ALS/SASRec이 이미 Implicit feature로 학습
- 명시적 Category boost는 불필요

### 4. **nDCG는 정확도 지표**
- Diversity는 도움 안 됨
- Top-3에 최고의 아이템 배치가 핵심

---

## 🎯 다음 전략 방향

### 전략 A: Phase 1 미세 조정 (보수적) ⭐⭐⭐⭐⭐
**기대 효과**: 0.1330 → 0.135~0.14 (+1.5~5%)

#### 접근
1. Phase 1 베이스 유지 (검증된 boost 값)
2. **ALS/SASRec 가중치 최적화**
   - 현재: 0.7 (ALS) / 0.3 (SASRec)
   - 시도: 0.75/0.25, 0.65/0.35, 0.8/0.2
3. **Recency 세밀화**
   - 1시간: 2.0배
   - 6시간: 1.7배 (NEW)
   - 12시간: 1.5배 (NEW)
   - 24시간: 1.3배 (조정)

#### 왜 이것인가?
- Phase 1이 검증됨 (0.1330)
- 작은 변화로 안전하게 개선
- 앙상블 가중치는 큰 영향력

---

### 전략 B: 완전히 다른 접근 - Session Boost ⭐⭐⭐⭐
**기대 효과**: 0.1330 → 0.135~0.142 (+1.5~9%)

#### 핵심 아이디어
- **48초 중앙값** 활용
- 짧은 시간에 여러 번 본 아이템 = 강한 관심
- Session burst 패턴 인식

#### 구현
```python
# Session 내 빈도 계산
# 5분 이내 상호작용을 1개 세션으로 묶음
session_count = count_interactions_in_5min_windows()

if session_count >= 3:  # 한 세션 내 3번
    boost *= 2.0
elif session_count >= 2:
    boost *= 1.5
```

---

### 전략 C: Ensemble Weight Grid Search ⭐⭐⭐
**기대 효과**: 0.1330 → 0.135~0.138 (+1.5~6%)

#### 접근
- Phase 1 베이스 고정
- ALS/SASRec 가중치만 변경
- 여러 조합 빠르게 생성

#### 테스트할 조합
```python
weights = [
    (0.75, 0.25),  # ALS 더 강조
    (0.65, 0.35),  # SASRec 더 강조
    (0.8, 0.2),    # ALS 매우 강조
    (0.6, 0.4),    # SASRec 많이 강조
]
```

---

## 🚀 권장 실행 순서

### 1단계: Ensemble Weight 최적화 (빠름, 안전)
- Phase 1 베이스 + 다양한 가중치
- 5개 조합 × 2분 = 10분
- 가장 빠르게 개선 가능

### 2단계: Recency 세밀화 (중간)
- 최적 가중치로 고정
- Recency window를 더 세분화
- 1~2% 추가 개선 기대

### 3단계: Session Boost (고급)
- 시간이 더 걸리지만 효과적
- 48초 패턴 활용
- 최종 돌파구

---

## 📈 예상 성능 로드맵

```
0.1330 (Phase 1) 
  ↓ +1~2% (Ensemble Weight)
0.1343~0.1356
  ↓ +1~2% (Recency 세밀화)  
0.1356~0.1383
  ↓ +2~4% (Session Boost)
0.1383~0.1438 (목표 0.14+)
```

---

## ⚠️ 피해야 할 것

1. ❌ Boost 값을 2.5 이상 올리지 말 것
2. ❌ Position-aware 같은 복잡한 로직
3. ❌ +100 같은 극단적 보너스
4. ❌ Category 기반 전략 (이미 실패)
5. ❌ Diversity 전략 (nDCG에 불리)

---

## ✅ 해야 할 것

1. ✅ Phase 1 베이스 유지
2. ✅ 작은 변화로 점진적 개선
3. ✅ 앙상블 가중치 최적화 우선
4. ✅ 검증된 boost 값 사용 (2.0~2.5 범위)
5. ✅ 빠른 실험 → 즉시 검증

---

## 🎯 결론

**Phase 1 (0.1330)이 현재 최고 성능**
- Event Type, Recency, Repeat의 균형 잡힌 조합
- 이것을 기준으로 미세 조정이 최선

**다음 액션**
1. Ensemble weight 최적화 (가장 빠르고 안전)
2. 최적 가중치로 recency 세밀화
3. 시간 있으면 session boost 시도
