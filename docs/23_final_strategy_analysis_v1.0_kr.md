# Phase 2-7 실패 분석 및 최종 전략

## 📊 전체 결과 요약

| Phase | nDCG@10 | 변화 | 전략 | Unique Items | Phase 1 Overlap |
|-------|---------|------|------|--------------|-----------------|
| **Phase 1** | **0.1330** | **Best** | Event + Recency + Repeat | 14,423 | - |
| Phase 2 | 0.1331 | +0.08% | Category + Diversity | 15,282 | 10/10 |
| Phase 3 | 0.1322 | -0.6% | Strong Boost | 14,379 | 10/10 |
| Phase 4 | 0.1310 | -1.5% | Weight 0.75/0.25 | 13,912 | 9/10 |
| Phase 5 | 0.1320 | -0.75% | Fine Recency | 14,604 | 9-10/10 |
| **Phase 6** | **?** | **?** | **ALS Only + Simple** | **2,519** | **7-8/10** |
| Phase 7 | ? | ? | Conservative Boost | 13,648 | 10/10 |

---

## 🔍 핵심 발견

### 1. **Phase 1이 Local Optimum**
- 모든 변경이 성능 저하
- Event Type (1.8x), Recency (2.0x), Repeat (2.5x)의 조합이 최적
- 더 강하게 → 실패
- 더 약하게 → 검증 필요
- 가중치 변경 → 실패

### 2. **Post-Processing의 한계**
- Phase 1 이후 post-processing으로는 개선 불가능
- Category, Diversity, Position-aware 모두 무의미
- 기본 모델(ALS/SASRec) 자체의 한계일 수 있음

### 3. **두 가지 남은 방향**

#### A. Phase 6 Simple (High Risk, High Return)
- **완전히 다른 접근**: ALS only + Basic Repeat
- **Unique items 2,519개** → 재방문에 극도로 집중
- Phase 1과 7-8/10만 겹침 → 완전히 다른 추천
- **장점**: 단순함, 재방문 집중, SASRec 노이즈 제거
- **단점**: Diversity 매우 낮음, 위험함

#### B. Phase 7 Conservative (Low Risk, Low Return)
- **보수적 접근**: Phase 1보다 약한 boost
- **Unique items 13,648개** → Phase 1과 유사
- Phase 1과 10/10 겹침 → 거의 동일
- **장점**: 안전함, Over-fitting 회피 가능
- **단점**: Phase 1과 너무 유사하여 큰 차이 없을 가능성

---

## 🎯 제출 전략

### 전략 1: Phase 6 Simple 먼저 시도 (추천) ⭐⭐⭐⭐⭐

**이유:**
1. **Phase 1과 완전히 다름** → 새로운 가능성
2. **재방문 집중** → nDCG는 정확도 지표, 재방문이 강력한 시그널
3. **SASRec 제거** → SASRec이 오히려 노이즈일 가능성
4. **단순함** → 복잡한 것보다 단순한 것이 더 나을 수 있음

**위험:**
- Diversity 낮음 (2,519 unique items)
- 크게 실패할 수도 있음 (0.12 이하 가능)

**예상:**
- 성공: 0.135~0.14 (+2~5%)
- 실패: 0.12~0.125 (-5~7%)

---

### 전략 2: Phase 7 Conservative (안전한 선택)

**이유:**
1. **Phase 1과 유사** → 안전함
2. **Over-fitting 회피** → Generalization 향상 가능
3. **점진적 개선** → 작은 개선 기대

**예상:**
- 최선: 0.1335~0.134 (+0.4~0.8%)
- 최악: 0.1315~0.132 (-0.8~1.1%)

---

## 💡 추천 순서

### 1단계: **Phase 6 Simple 제출** ✅
- High risk, high return
- 완전히 다른 접근이므로 시도할 가치 있음
- 실패하더라도 인사이트 얻을 수 있음

### 2단계: Phase 6 결과에 따라

#### If Phase 6 > 0.1330 (성공)
→ **Phase 6 기반 개선**
- ALS only 유지
- Repeat boost 미세 조정
- Event Type 추가 (purchase만)

#### If 0.125 < Phase 6 < 0.1330 (중간)
→ **Phase 7 Conservative 제출**
- 안전한 선택으로 복귀
- Phase 1 대비 작은 개선 기대

#### If Phase 6 < 0.125 (대실패)
→ **근본적 재검토 필요**
- ALS/SASRec 모델 자체의 문제
- 하이퍼파라미터 재튜닝 필요
- 또는 다른 모델 시도 (EASE, 규칙 기반 등)

---

## 📈 Phase 6 Simple 상세 분석

### 특징
```python
# Phase 6 Simple 전략
Model: ALS only (no SASRec)
Boost: Repeat only (no Event/Recency)
Values: 1.3/2.0/2.5 (count 1/2/3+)
```

### 왜 성공할 수 있나?

1. **재방문의 힘**
   - 18.3% 재방문률
   - 재방문은 가장 강력한 시그널
   - nDCG는 정확도 지표 → 확실한 것에 집중

2. **단순함의 미학**
   - Phase 1은 9가지 feature 조합 (복잡)
   - Phase 6은 1가지 feature (단순)
   - "Simple is better than complex" (Python Zen)

3. **SASRec 노이즈 제거**
   - SASRec은 Sequential model
   - 이 데이터에서는 Co-occurrence(ALS)가 더 효과적일 수 있음
   - 0.3 가중치가 오히려 방해일 가능성

4. **Diversity의 역설**
   - Phase 6: 2,519 unique items (낮은 diversity)
   - Phase 1: 14,423 unique items (높은 diversity)
   - nDCG는 정확도 중심 → diversity 필요 없음
   - 확실한 것 10개만 추천하는 게 나을 수 있음

---

## ⚠️ 만약 Phase 6/7도 실패한다면?

### Plan B: 근본적 접근 전환

1. **모델 개선**
   - ALS 하이퍼파라미터 재튜닝 (factors, regularization, iterations)
   - SASRec 하이퍼파라미터 재튜닝 (hidden_size, num_layers, dropout)
   - 다른 모델 시도 (EASE, ItemKNN, BPR)

2. **Feature Engineering**
   - Time-of-day features
   - User/Item popularity
   - Co-occurrence patterns
   - Session-based features

3. **Ensemble 전략**
   - RRF (Reciprocal Rank Fusion)
   - Borda count
   - Learning to Rank

4. **규칙 기반**
   - Top-N에서 N을 늘리고 (예: Top-100)
   - 규칙으로 Top-10 선택
   - Hybrid approach

---

## 🎯 최종 결론

**Phase 6 Simple을 시도하세요!**

- ✅ 완전히 다른 접근
- ✅ 재방문 집중 (가장 강력한 시그널)
- ✅ 단순함 (복잡성 제거)
- ✅ High risk, high return
- ✅ 인사이트 확보 (실패해도 배울 것 많음)

**Phase 1 (0.1330)이 최종 답일 수도 있습니다.**
- Post-processing의 한계
- 기본 모델의 한계
- 더 이상의 개선은 모델 자체 개선 필요

**하지만 시도해볼 가치는 충분합니다!** 🚀
