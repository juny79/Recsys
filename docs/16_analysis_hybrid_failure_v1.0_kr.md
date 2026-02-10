# Hybrid Ensemble 실패 분석 및 개선 제안

## 1. 실패 원인 분석 (Score 0.1180 < 0.1196)
- **과도한 XGBoost 의존:** Short User에게 XGBoost 가중치를 **0.5**나 준 것이 패착입니다.
    - XGBoost 단독 점수(0.0853)는 ALS(0.1124)에 비해 너무 낮습니다.
    - 아무리 Cold Start라도, 성능 차이가 큰 모델에게 메인 키를 쥐어주는 것은 위험했습니다.
- **결론:** XGBoost는 **"보조(0.15)"**로 쓸 때 가장 빛납니다. 주력으로 격상시키면 전체 성능을 깎아먹습니다.

## 2. 새로운 전략: EASE^R (Embarrassingly Shallow Autoencoders)
XGBoost 말고, **ALS와 다른 방식으로 문제를 푸는** 강력한 모델이 필요합니다.

### 🔍 EASE^R 추천 이유
1.  **SOTA급 성능:** 희소한 데이터셋(Sparse Data)에서 ALS나 심지어 딥러닝 모델보다 좋은 성능을 자주 냅니다.
2.  **완전 다른 메커니즘:**
    - **ALS:** Matrix Factorization (Latent Space)
    - **SASRec:** Sequential (Time-series)
    - **EASE^R:** **Linear Autoencoder (Item-Item Weights)**
3.  **앙상블 효과 극대화:** ALS와 메커니즘이 다르므로, 앙상블 시 상호 보완 효과가 큽니다.

### 3. 실행 계획
1.  **EASE^R 구현 (`train_ease.py`)**: 학습 및 추론 (매우 빠름).
2.  **4-Way Ensemble**: ALS + SASRec + XGB + **EASE**.
