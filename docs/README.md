# 문서 관리 가이드

본 디렉토리는 커머스 추천 시스템 프로젝트의 모든 보고서와 문서를 관리합니다.

## 📁 디렉토리 구조

```
docs/
├── README.md (본 파일)
├── 01_implementation_plan_v1.0_kr.md
└── 02_completion_report_v1.0_kr.md
```

## 📋 문서 목록

| 파일명 | 버전 | 작성일 | 설명 |
|--------|------|--------|------|
| `01_implementation_plan_v1.0_kr.md` | v1.0 | 2026-02-06 | ALS 베이스라인 개선 구현 계획서 |
| `02_completion_report_v1.0_kr.md` | v1.0 | 2026-02-06 | 이벤트 가중치 & 시간 감쇠 구현 완료 보고서 |
| `03_ndcg_score_analysis_v1.0_kr.md` | v1.0 | 2026-02-06 | NDCG@10 점수 분석 및 해석 가이드 (현재: 0.0955) |
| `04_sasrec_implementation_plan_v1.0_kr.md` | v1.0 | 2026-02-06 | SASRec 모델 개선 구현 계획서 |
| `05_sasrec_failure_analysis_v1.0_kr.md` | v1.0 | 2026-02-06 | ⚠️ SASRec 성능 저하 원인 분석 보고서 (NDCG: 0.0843) |
| `06_sasrec_als_comparison_v1.0_kr.md` | v1.0 | 2026-02-07 | 📊 SASRec vs ALS 최종 비교 분석 보고서 |
| `07_als_tuning_strategy_v1.0_kr.md` | v1.0 | 2026-02-07 | 🔧 ALS 하이퍼파라미터 튜닝 전략 |

## 📝 문서 명명 규칙

새로운 문서를 추가할 때는 다음 형식을 따라주세요:

```
<번호>_<문서명>_v<버전>_kr.md
```

**예시:**
- `01_implementation_plan_v1.0_kr.md` - 구현 계획서
- `02_completion_report_v1.0_kr.md` - 완료 보고서
- `03_experiment_results_v1.0_kr.md` - 실험 결과 보고서

## 🔄 버전 관리

- **v1.0**: 초기 버전
- **v1.1**: 마이너 수정 (오타, 형식 수정 등)
- **v2.0**: 메이저 변경 (내용 대폭 수정, 재작성 등)

## 📌 문서 유형

### 계획서 (Plan)
- 구현 전 작성하는 설계 및 계획 문서
- 예: `implementation_plan`, `design_proposal`

### 보고서 (Report)
- 작업 완료 후 결과를 정리한 문서
- 예: `completion_report`, `experiment_results`

### 분석서 (Analysis)
- 데이터 분석 또는 성능 분석 문서
- 예: `data_analysis`, `performance_analysis`

---

**최종 업데이트:** 2026-02-06
