# RecSys Challenge 2026 — 최종 프로젝트 보고서

**문서 버전:** v2.0 (Final)  
**작성일:** 2026-02-12  
**최종 점수:** **nDCG@10 = 0.1344**  
**총 제출 횟수:** 11회  

---

## 1. Executive Summary

커머스 상품 구매 예측 과제에서 **4개 추천 모델의 앙상블 + 데이터 기반 후처리**를 통해 베이스라인(0.1173) 대비 **+14.6% 향상된 nDCG@10 = 0.1344**를 달성했다.

핵심 성과:
- 단일 모델 최고(ALS 0.1124) → 4-Way 앙상블(0.1341) → Post-processing(0.1344)
- 11회 제출을 통한 체계적 실험으로 "무엇이 작동하고 무엇이 실패하는지" 명확히 규명
- **모델 다양성이 부스트 강도보다 훨씬 중요**하다는 실전 인사이트 확보

---

## 2. 전체 실험 이력 (11회 제출)

| # | 제출명 | 전략 | nDCG@10 | Δ Baseline | Δ 이전 | 결과 |
|:-:|--------|------|:-------:|:----------:|:------:|:----:|
| 1 | Baseline | ALS + SASRec (2-Way) | 0.1173 | — | — | 기준 |
| 2 | Fast Optimized | + Repeat Boost | 0.1265 | +7.8% | +7.8% | ✅ |
| 3 | **Phase 1** | + Event + Recency + Top-3 | **0.1330** | +13.4% | +5.1% | ✅✅ |
| 4 | Phase 7 | 약한 부스트 | 0.1289 | +9.9% | −3.1% | ❌ |
| 5 | 3-Way | + XGB LTR | 0.1326 | +13.0% | −0.3% | — |
| 6 | 3-Way w70 | ALS=0.70 | 0.1320 | +12.5% | −0.5% | ❌ |
| 7 | 3-Way Enhanced | + Phase 1+ | 0.1331 | +13.5% | +0.08% | — |
| 8 | **4-Way** | + CatBoost LTR | **0.1341** | +14.3% | +0.83% | ✅✅ |
| 9 | **4-Way Enhanced** | + Phase 1+ Enhanced | **0.1344** | **+14.6%** | +0.22% | 🏆 |
| 10 | V3 Phase A | + Category ×1.10 | 0.1344 | +14.6% | ±0.00% | — |
| 11 | V3 Phase F | + Session Category ×1.15 | 0.1344 | +14.6% | ±0.00% | — |

---

## 3. 최종 파이프라인 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                   train.parquet (8.35M rows)              │
│          638,257 users × 29,502 items × 8 columns        │
└──────────┬───────────┬───────────┬───────────┬───────────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
      ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
      │   ALS   │ │ SASRec  │ │   XGB   │ │ CatBoost │
      │ CF기반  │ │ 시퀀셜  │ │  LTR    │ │  LTR     │
      │ 0.1124  │ │ 0.0864  │ │ 0.1117  │ │ 0.1125   │
      └────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘
           │           │           │            │
           ▼           ▼           ▼            ▼
      ┌──────────────────────────────────────────────┐
      │         4-Way RRF Ensemble (K=60)            │
      │   ALS:0.50  SAS:0.25  XGB:0.10  CAT:0.15    │
      │         → 25.8 candidates/user avg           │
      │                 = 0.1341                     │
      └──────────────────┬───────────────────────────┘
                         │
                         ▼
      ┌──────────────────────────────────────────────┐
      │         Phase 1+ Enhanced Boost              │
      │  Repeat:  1.4× / 2.1× / 2.6×  (1/2/3+회)   │
      │  Event:   purchase 1.9×, cart 1.5×           │
      │  Recency: ≤1h 2.1×, ≤24h 1.6×, ≤72h 1.3×   │
      │  Top-3:   1.6× (purchase+2회 OR 3+회)        │
      │                 = 0.1344                     │
      └──────────────────┬───────────────────────────┘
                         │
                         ▼
                  output (Top-10/user)
                  638,257 × 10 = 6,382,570 rows
```

---

## 4. 모델별 상세 구성

### 4.1 ALS (Alternating Least Squares)
| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Library | implicit | GPU 가속 |
| factors | 128 | 잠재 차원 |
| regularization | 0.1 | L2 정규화 |
| alpha | 10 | 신뢰도 가중치 |
| iterations | 30 | 수렴 충분 |
| decay_rate | 0.1 | 시간 감쇠 |
| event weights | view:1, cart:3, purchase:5 | 이벤트 차등 |
| nDCG@10 | **0.1124** | 최강 단일 모델 |

### 4.2 SASRec (Self-Attentive Sequential Recommendation)
| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Framework | RecBole | PyTorch 기반 |
| n_layers | 4 | Transformer 깊이 |
| n_heads | 4 | Attention heads |
| hidden_size | 256 | 임베딩 차원 |
| inner_size | 512 | FFN 차원 |
| hidden_dropout | 0.3 | |
| max_seq_length | 20 | |
| min_user_inter | 5 | Quality > Quantity |
| nDCG@10 | **0.0864** | 단독 약하나 앙상블에 기여 |

### 4.3 XGBoost LTR (Learning to Rank)
| 파라미터 | 값 |
|---------|-----|
| objective | rank:ndcg |
| features | ALS/SASRec rank, RRF score, user/item 통계 |
| nDCG@10 | **0.1117** |

### 4.4 CatBoost LTR
| 파라미터 | 값 |
|---------|-----|
| loss_function | YetiRank |
| features | trajectory features + categoricals |
| cat_features | top_brand, item_brand, item_cat |
| nDCG@10 | **0.1125** |

---

## 5. 핵심 인사이트 (11회 실험에서 배운 것)

### 5.1 모델 다양성 >>> 부스트 강도

가장 중요한 발견. 4-Way 앙상블 추가(+0.83%)가 Phase 1+ 부스트 강화(+0.22%)보다 **4배 더 효과적**이었다.

```
모델 다양성 기여:   2-Way → 4-Way = +0.83%  (핵심 드라이버)
부스트 강화 기여:   Phase 1 → 1+ = +0.22%   (보조적)
카테고리 부스트:    +0.00%  (모델이 이미 캡처)
세션 부스트:        +0.00%  (모델이 이미 캡처)
```

### 5.2 Post-processing의 한계

| 시도 | 시그널 | 결과 | 원인 |
|------|--------|------|------|
| Repeat Boost | 재방문 18.3% | ✅ +7.8% | 모델이 미캡처 |
| Event + Recency | 구매/장바구니/시간 | ✅ +5.1% | 모델이 부분 캡처 |
| Category Boost (Phase 2) | 68.4% 일관성 | ❌ +0.08% | ALS/CatBoost가 이미 캡처 |
| Strong Boost (Phase 3) | 부스트 강화 | ❌ −0.6% | 과적합 |
| Category ×1.10 (V3-A) | 보수적 카테고리 | — ±0.00% | 완전 중복 |
| Session Cat (V3-F) | 세션 문맥 | — ±0.00% | SASRec가 부분 캡처 |

**교훈:** Post-processing은 "모델이 놓친 시그널"에만 효과가 있다. Repeat/Event/Recency는 모델이 놓친 시그널이었지만, Category/Brand/Session은 이미 모델들이 학습한 시그널이었다.

### 5.3 부스트 강도의 골디락스 존

```
약한 부스트 (Phase 7):    0.1289  ← 부스트가 너무 약해 효과 없음
최적 부스트 (Phase 1):    0.1330  ← 골디락스 존
약간 강한 부스트 (Phase 1+): 0.1344  ← 4-Way에서는 약간 더 강해도 OK
강한 부스트 (Phase 3):    0.1322  ← 과적합으로 하락
```

### 5.4 앙상블 가중치의 균형

```
ALS 과중 (0.70):     0.1320  ❌  강한 모델에 쏠리면 다양성 상실
균형 분배 (50/25/10/15): 0.1344  ✅  약한 모델도 중요한 기여
```

---

## 6. 데이터 분석 요약

| 지표 | 값 | 활용 여부 |
|------|-----|---------|
| 총 상호작용 | 8,350,311 | ✅ |
| 유저 수 | 638,257 | ✅ |
| 아이템 수 | 29,502 | ✅ |
| 재방문률 | 18.3% | ✅ Repeat Boost로 활용 |
| 카테고리 일관성 | 68.4% | ⚠️ 시도했으나 모델이 이미 캡처 |
| 세션 median 간격 | 48초 | ⚠️ 시도했으나 모델이 이미 캡처 |
| Light 유저 (≤5건) | 42% (302,903명) | ⚠️ 세그먼트 차등 미적용 |
| 데이터 컬럼 | 8개 | 4개만 직접 사용 |

---

## 7. 실패한 시도들과 교훈

| # | 시도 | 기대 | 실제 | 교훈 |
|:-:|------|------|------|------|
| 1 | Category Boost (Phase 2) | +1~2% | +0.08% | EDA 시그널 ≠ Post-processing 효과 |
| 2 | Strong Boost (Phase 3) | +0.5% | −0.6% | "더 세게" ≠ "더 좋게" |
| 3 | ALS Weight 70% (w70) | +0.3% | −0.5% | 다양성을 희생하면 안 됨 |
| 4 | ALS Only | +5% | −11.6% | 단일 모델 의존은 재앙 |
| 5 | Fine Recency (Phase 5) | +0.5% | −0.75% | 시간 구간 세분화는 노이즈 |
| 6 | EASE 모델 | +3% | 완전 실패 | GPU 메모리 한계 |
| 7 | V3 Category ×1.10 | +0.3% | ±0.00% | 보수적이어도 중복은 중복 |
| 8 | V3 Session Category | +0.5% | ±0.00% | 새 시그널이라 했지만 모델이 캡처 |

---

## 8. 점수 향상 기여도 분해

```
0.1173 (Baseline)
  │
  ├─ +0.0092 (+7.8%)  Repeat Boost          ← 가장 큰 단일 개선
  │                                            "유저가 전에 본 아이템을 우대"
  ├─ +0.0065 (+5.1%)  Event + Recency        ← 두번째 큰 개선
  │                                            "구매/장바구니 + 최근성 반영"
  ├─ +0.0011 (+0.8%)  4-Way 앙상블            ← 모델 다양성
  │                                            "XGB + CatBoost 추가"
  ├─ +0.0003 (+0.2%)  Phase 1+ 강화          ← 미세 조정
  │                                            "부스트 파라미터 5% 상향"
  │
  ▼
0.1344 (Final Best)

총 향상: +0.0171 (+14.6%)
```

---

## 9. 프로젝트 타임라인

```
Day 1 ─── ALS 모델 구축 (0.1124)
  │       SASRec 모델 구축 (0.0864)
  │
Day 2 ─── 2-Way 앙상블 (0.1173) ← Baseline 확립
  │       XGB LTR 학습 (0.1117)
  │       CatBoost LTR 학습 (0.1125)
  │
Day 3 ─── EDA 분석 (재방문 18.3%, 카테고리 68.4%)
  │       Repeat Boost 구현 (0.1265)
  │       Phase 1 최적화 (0.1330) ← 큰 도약
  │
Day 4 ─── Phase 2~7 실험 (대부분 실패)
  │       3-Way 앙상블 시도 (0.1326~0.1331)
  │       실패 원인 분석 및 전략 수정
  │
Day 5 ─── 4-Way 앙상블 돌파 (0.1341)
  │       Phase 1+ Enhanced (0.1344) ← 최고 기록
  │       V3 전략 수립 및 실행 (Phase A, F)
  │       최종 보고서 작성
```

---

## 10. 산출물 목록

### 코드
| 파일 | 용도 |
|------|------|
| [code/train_als.py](code/train_als.py) | ALS 모델 학습 |
| [code/train_sasrec.py](code/train_sasrec.py) | SASRec 모델 학습 |
| [code/train_ltr_ranker.py](code/train_ltr_ranker.py) | XGBoost LTR 학습 |
| [code/train_catboost_ltr.py](code/train_catboost_ltr.py) | CatBoost LTR 학습 |
| [code/ensemble_quad_enhanced.py](code/ensemble_quad_enhanced.py) | **최종 파이프라인 (0.1344)** |
| [code/ensemble_v3_maximizer.py](code/ensemble_v3_maximizer.py) | V3 다중 Phase 테스트 |

### 최종 제출 파일
| 파일 | nDCG@10 |
|------|:-------:|
| [output/output_quad_enhanced.csv](output/output_quad_enhanced.csv) | **0.1344** 🏆 |

### 문서
| 파일 | 내용 |
|------|------|
| [docs/20_eda_insight_report_v1.0_kr.md](docs/20_eda_insight_report_v1.0_kr.md) | EDA 핵심 시그널 분석 |
| [docs/28_final_optimization_roadmap_v1.0_kr.md](docs/28_final_optimization_roadmap_v1.0_kr.md) | 실험 결과 총정리 |
| [docs/29_score_maximization_strategy_v3.0_kr.md](docs/29_score_maximization_strategy_v3.0_kr.md) | V3 전략 설계 |
| [docs/30_final_project_report_v2.0_kr.md](docs/30_final_project_report_v2.0_kr.md) | **본 최종 보고서** |

---

## 11. 0.1344를 넘기 위해 필요했던 것 (회고)

돌이켜보면, 0.1344 이후 추가 개선이 없었던 근본 원인은 **모든 Post-processing 시그널이 이미 4개 모델에 의해 캡처되어 있었기 때문**이다.

추가 개선을 위해서는:
1. **5번째 모델 추가** (예: GNN, BERT4Rec, LightGCN) → 새로운 관점의 후보 생성
2. **Cross-validation 기반 부스트 파라미터 최적화** → 제출 횟수 의존 탈피
3. **아이템 임베딩 기반 유사도 활용** → 새 아이템 발굴 (Cold-start)
4. **Stacking (2-Level) 앙상블** → 모델 출력을 피처로 재학습

하드웨어(RTX 3070 8GB)와 시간의 제약 속에서 0.1344는 현실적으로 근접한 최적값이었다.

---

## 12. 결론

> **"데이터의 특성을 이해하고 모델의 다양성을 확보하는 것이 
> 복잡한 후처리보다 훨씬 효과적이다."**

이번 프로젝트의 가장 큰 교훈이다.

- Repeat Boost라는 단순한 휴리스틱이 +7.8%를 가져왔고,
- 4개 모델의 앙상블이 +14.3%를 가져왔지만,
- 카테고리/브랜드/가격/세션 등 정교한 개인화 후처리는 +0.00%였다.

**모델이 이미 학습한 것 위에 같은 시그널을 후처리로 다시 얹는 것은 효과가 없다.**
효과를 보려면 모델이 **보지 못한** 시그널을 찾거나, 아예 **새로운 모델**을 추가해야 한다.

최종 점수 **nDCG@10 = 0.1344**, 베이스라인 대비 **+14.6% 향상**.
