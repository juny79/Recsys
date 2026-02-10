# 최종 프로젝트 리포트: RecSys Challenge 2026

**문서 버전:** v1.0  
**작성일:** 2026-02-09  
**최종 점수:** **NDCG@10 = 0.1173** (Rank 1st Estimate)

---

## 1. 프로젝트 여정 요약 (Executive Summary)
본 프로젝트는 **SASRec(Sequential)**과 **ALS(Collaborative Filtering)**의 강점을 결합하여 추천 성능을 극대화하는 과정을 거쳤습니다. 초기에는 SASRec의 성능 저하로 난항을 겪었으나, **"데이터 필터링 전략"**과 **"단순 가중 앙상블"**의 힘을 입증하며 최종적으로 **0.1173**이라는 최고 점수를 달성했습니다.

| 단계 | 전략 | 점수 (NDCG@10) | 비고 |
|---|---|---|---|
| **Phase 1** | ALS Tuning (Decay 0.1) | 0.1124 | 강력한 베이스라인 확립 |
| **Phase 2** | SASRec (Initial) | 0.0855 | 데이터 부족으로 성능 한계 |
| **Phase 3** | SASRec (Relaxed Filter) | 0.0813 (▼) | Short User 포함이 오히려 노이즈가 됨 |
| **Phase 4** | Ensemble (ALS+SASRec) | 0.1150 | 상호 보완 효과 확인 |
| **Phase 5** | Advanced Ensemble (Segmented) | 0.1100 (▼) | SASRec의 Context 무시가 뼈아픔 |
| **Phase 6** | RRF Ensemble (K=60) | 0.1103 (▼) | ALS의 강력한 신호 희석 |
| **Phase 7** | **SASRec Improved** | **0.0864** (Old 0.0813) | **Strict Filter + Deep Arch**로 성능 복구 |
| **Final** | **Ensemble (Verified)** | **0.1173 (Best)** | **Best ALS + Best SASRec** |

---

## 2. 주요 실험 및 인사이트

### 2.1 SASRec의 부활: "Quantity < Quality"
- **실패:** 데이터가 부족하다고 판단하여 필터링 기준을 완화(`2개 이상`)하고 Short User를 포함시켰으나, 성능은 오히려 **0.0855 -> 0.0813**으로 떨어졌습니다.
- **성공:** 다시 **5개 이상**의 High Quality 유저만 남기고, 대신 모델의 깊이(Layer 4)와 너비(Head 4)를 키웠더니 **0.0864**로 자체 최고점을 경신했습니다.
- **교훈:** 노이즈가 많은 데이터를 억지로 학습시키는 것보다, 양질의 데이터로 패턴을 확실히 학습시키는 것이 유리합니다.

### 2.2 앙상블 전략: "Simple is Best"
- **실패 (Segmentation):** 유저 길이에 따라 모델 비중을 다르게 가져가는 복잡한 전략은 실패했습니다(0.1100). 짦은 시퀀스에서도 SASRec의 '직전 아이템' 정보는 중요했기 때문입니다.
- **실패 (RRF):** 1등과 2등의 점수 차를 줄이는 RRF(Reciprocal Rank Fusion) 방식은 ALS의 강력한 확신을 희석시켜 성능을 떨어뜨렸습니다(0.1103).
- **성공 (Weighted Sum):** 가장 단순한 **순위 역수 합산 (`1/rank`)** 방식이 가장 좋았습니다. 잘하는 모델(ALS)에 0.7, 보조 모델(SASRec)에 0.3을 주는 고전적인 방식이 결국 정답이었습니다.

---

## 3. 최종 산출물
- **코드:**
    - `code/train_sasrec.py`: Tuned Hyperparameters (L=4, H=4, Drop=0.3)
    - `code/ensemble.py`: Weighted Rank Ensemble Logic
- **데이터:**
    - `output/output_ensemble_final.csv`: 최종 제출 파일 (NDCG 0.1173)

---

## 4. 맺음말
이번 프로젝트는 **"데이터의 품질"**과 **"모델의 조화"**가 얼마나 중요한지 보여주었습니다. 무조건 최신 기법(RRF)이나 복잡한 전략(Segmentation)을 쓰는 것보다, 데이터의 특성을 이해하고 기본기를 충실히 다지는 것이 최고의 성능을 낸다는 것을 다시 한번 확인했습니다.

**수고 많으셨습니다!**
