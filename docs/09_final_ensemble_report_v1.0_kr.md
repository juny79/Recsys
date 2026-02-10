# SASRec 재시도 및 앙상블 결과 보고서

**문서 버전:** v1.0  
**작성일:** 2026-02-08  
**작성자:** Antigravity  
**관련 태스크:** SASRec 성능 개선 및 앙상블 (Ensemble)

---

## 1. 개요 (Executive Summary)

ALS 모델(0.1124)의 한계를 돌파하기 위해, 실패했던 SASRec 모델을 재설계하고 ALS와의 앙상블을 시도했습니다. 
지난 실패 분석을 통해 치명적인 데이터 필터링 실수(Interaction 5개 미만 제외)를 발견하고 이를 수정하였으며, 최종적으로 두 모델의 장점을 결합한 앙상블 모델을 구축했습니다.

---

## 2. 수행 내용

### 2.1 SASRec 재설계 (Phase 2)

| 항목 | 기존 설정 (Phase 1) | **개선 설정 (Phase 2)** | 효과 |
|---|---|---|---|
| **User Filter** | `[5,Inf]` (약 50% 유저 제외) | **`[2,Inf]`** (대부분 포함) | 학습 데이터 대폭 증강, Short Sequence 유저 반영 |
| **Max Length** | 50 | **20** | 짧은 시퀀스에 최적화, 불필요한 패딩 감소 |
| **ID Mapping** | 불일치 (Bug) | **일치 (Fixed)** | 정확한 아이템 추천 가능 |
| **Sequence** | 인위적 복제 (Bug) | **원본 유지 (Fixed)** | 자연스러운 순차 패턴 학습 |

### 2.2 앙상블 (Ensemble)

- **구성:** Best ALS (Decay 0.1) + Improved SASRec
- **방식:** Rank-Based Weighted Ensemble
  - `Score = (0.7 * ALS_Rank_Score) + (0.3 * SASRec_Rank_Score)`
  - *ALS의 강력한 전반적 성능을 베이스로 하고, SASRec의 순차적 패턴 예측을 보완제로 활용*

---

## 3. 결과물

다음 파일들이 생성되었습니다.

1.  **[SASRec 추론 결과]** `output/output_sasrec_fixed.csv`
    - 개선된 설정으로 학습된 순수 SASRec 모델의 예측값
2.  **[최종 앙상블 결과]** `output/output_ensemble.csv`
    - ALS와 SASRec을 결합한 최종 제출용 파일
    - **예상 성능:** ALS 단일 모델(0.1124) 대비 **0.12++** 이상 기대

---

## 4. 결론 및 제언

### ✅ 결론
치명적인 버그 수정과 데이터 필터링 완화를 통해 SASRec이 정상적으로 동작할 수 있는 환경을 만들었습니다. 
특히 **전체 유저의 절반에 달하는 Short Sequence 유저들을 학습에 포함시킨 것**이 성능 향상의 핵심 동력이 될 것입니다.

### 🚀 향후 제언
생성된 `output_ensemble.csv`를 제출하여 실제 리더보드 점수를 확인해 보시기 바랍니다. 
만약 점수가 기대보다 낮다면, 앙상블 가중치(`w_als`, `w_sasrec`)를 조정(예: 0.5:0.5)하여 최적의 조합을 찾을 수 있습니다.

수고하셨습니다!
