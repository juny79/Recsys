# [Fundamental Shift] Segmented Intent-Driven Re-ranking

## 1. 실패 분석: LTR (0.1117)이 왜 앙상블 (0.1196)보다 낮았는가?
- **데이터 희소성:** 학습 데이터(Lables)가 전체 유저의 극히 일부에 국한되어, 모델이 대다수 유저의 일반적인 경향성보다 특정 유저의 노이즈를 학습했을 가능성이 큽니다.
- **피처 부족:** 단순히 모델의 Rank만으로는 ALS의 'Time Decay'가 가진 강력한 시간성 신호를 넘어서지 못했습니다.
- **모델 일원화:** 활동량이 적은 유저(Cold)와 많은 유저(Heavy)를 하나의 모델로 처리하면서, 두 그룹의 서로 다른 추천 로직이 충돌했습니다.

## 2. 근본적 전환 전략: [Segmented CatBoost LTR]

### 핵심 1: 유저 세그먼트별 맞춤 모델 (Segmentation)
- **Lite User (이력 < 5건):** 아이템 간 유사도(Item-to-Item)와 카테고리 인기도(Popularity) 기반의 '강한 복구' 전략.
- **Heavy User (이력 >= 5건):** Transformer 기반 시퀀스 패턴(SASRec)과 브랜드 충성도(Brand Loyalty)를 극대화하는 LTR 전략.

### 핵심 2: 궤적 피처 (Trajectory Features) 도입
단순한 상태(State)가 아닌 **흐름(Trend)**을 피처화합니다.
- **Price Drift:** 유저가 최근에 보고 있는 아이템들의 가격이 점진적으로 상승/하락하고 있는가?
- **Category Entropy:** 유저가 한 우물(예: 가전)만 파고 있는가, 아니면 이것저것 구경 중인가?
- **Recency intensity:** 최근 1시간 이내의 활동이 평소보다 얼마나 집중되어 있는가?

### 핵심 3: CatBoost 모델로의 전환
- **Categorical Optimization:** 브랜드, 카테고리 코드가 수만 개인 본 데이터셋에서 XGBoost보다 CatBoost의 Ordered Boosting이 더 강력한 성능을 발휘합니다.
- **Robustness:** 하이퍼파라미터 튜닝 없이도 과적합에 강한 CatBoost의 특성을 활용합니다.

## 3. 향후 로드맵
1. **Trajectory 추출:** `features_v3.py`에서 유저의 흐름(Trend) 피처 생성.
2. **Segmented Labeling:** 최근 상호작용뿐만 아니라 '세션 내 재방문'을 레이블로 활용.
3. **CatBoost LTR 학습:** `train_catboost_ltr.py` 실행.

이 전략은 단순한 '합산'이 아닌, **데이터의 맥락(Context)을 이해하는 추천**으로의 전환을 의미합니다.
