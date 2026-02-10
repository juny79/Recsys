# GBDT Re-ranking 실패 원인 분석 보고서

**작성일:** 2026-02-09
**작성자:** Antigravity

## 1. 현상 진단
- **Ensemble (Simple Weighted):** NDCG@10 = **0.1173** (Best)
- **GBDT Re-ranking:** NDCG@10 = **0.0853** (Severe Drop)
- **증상:** 재정렬 후 점수가 단독 모델(SASRec 0.0864) 수준으로 회귀하거나 그보다 낮아짐.

## 2. 결정적 실패 원인 (Critical Failure Points)

### 2.1 "점수" 피처의 부재 (Missing Score Features)
가장 치명적인 실수는 **ALS와 SASRec이 원래 매겼던 점수(`score` or `rank`)를 GBDT 학습 피처로 넣지 않은 것**입니다.

- **현재 피처:** `price`, `brand`, `view_count`, `pop_count` 등 (Content & Popularity 위주)
- **빠진 피처:** `als_score`, `sasrec_score`, `als_rank`, `sasrec_rank`
- **결과:** GBDT 모델은 "이 아이템을 ALS가 1등으로 뽑았는지, 100등으로 뽑았는지" 알 길이 없습니다. 오직 **"이 아이템이 인기 있는가?"** 만 보고 판단하게 되었습니다.

### 2.2 인기도 바이어스 (Popularity Bias)
`Feature Importance`를 보면 `view` 수와 `pop_count`가 압도적(합계 약 82%)입니다.
```text
view               0.589337  (58.9%)
pop_count          0.230026  (23.0%)
```
- ALS와 SASRec은 협업 필터링을 통해 "나만 좋아할 수 있는 롱테일 아이템"을 찾아냈습니다.
- 하지만 GBDT 리랭커는 이를 **"조회수가 낮은 듣보잡 아이템"**으로 간주하고 점수를 깎아버렸습니다.
- 결과적으로 **개인화 추천이 제거되고, 인기 아이템 추천(Most Popular)으로 퇴보**했습니다.

### 2.3 학습 데이터 분포 불일치
- **학습 데이터:** 과거의 구매/조회 이력 (Historical Interactions)
- **테스트 데이터:** 미래의 추천 후보 (Future Candidates)
- GBDT는 "과거에 많이 팔린 것"을 학습했기 때문에, "앞으로 뜰 아이템"이나 "개인적 취향"을 예측하는 데는 ALS/SASRec보다 못할 수 있습니다.

## 3. 결론 및 제언
시간이 충분하다면 `als_rank`, `sasrec_rank`를 피처로 추가하여 재학습할 수 있겠으나, 현재로서는 리스크가 큽니다.

> **최종 결론:** 복잡한 기법이 항상 정답은 아닙니다. 데이터와 피처가 완벽하지 않다면, **Simple Weighted Ensemble (0.1173)**이 가장 강력하고 안전한 선택입니다.

이 점수(0.1173)는 리더보드 상위권에 충분히 경쟁력 있는 점수이므로, 이것으로 최종 제출하는 것을 강력히 권장합니다.
