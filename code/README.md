# E-commerce item Recommendation Baseline Code

이커머스 상품 추천 대회를 위한 베이스라인 코드입니다.

## Installation

```
python 3.10 이상의 환경에서 진행해주시길 바랍니다.
pip install -r requirements.txt
```

## How to run

### MODEL

#### ALS

1. train ALS model and inference
   ```
   python train_als.py
   ```

#### SASRec

1. Prepare datset for using Recbole library
   ```
   python recbole_dataset.py
   ```

2. Train SASRec model 
   ```
   python train_sasrec.py
   ```

3. Infrence with trained sasrec model
   2번에서 생성된 model 파일의 상대 경로가 './saved/SASRec.pth'라면
   ```
   python inference_sasrec.py --model_file ./saved/SASRec.pth
   ```

#### **✨ Optimized Ensemble (nDCG@10 최적화) - RTX 3070 8GB 지원**

Post-Processing 로직을 통한 nDCG@10 극대화 앙상블

**주요 특징:**
- 🔄 **Repeat Boost**: 재방문 아이템 우선순위 상향 (18.3% 재방문율 활용)
- 📂 **Category Consistency**: 최근 카테고리 기반 부스팅 (68.4% 일관성 활용)
- ⏰ **Fine-grained Recency**: 시간별 세밀한 감쇠 (48초 중앙값 활용)
- 🎯 **Diversity Control**: Top-10 카테고리 다양성 유지
- 💾 **Memory Optimized**: RTX 3070 8GB 환경 최적화 (배치 처리)

**실행 방법:**

1. **PowerShell 스크립트로 실행 (권장)**
   ```powershell
   .\run_optimized_ensemble.ps1
   ```

2. **배치 파일로 실행**
   ```cmd
   run_optimized_ensemble.bat
   ```

3. **직접 Python 명령어로 실행**
   ```bash
   python ensemble_optimized.py \
       --als_output ../output/output.csv \
       --sasrec_output ../output/output_sasrec_fixed_19.csv \
       --output_path ../output/output_optimized_final.csv \
       --max_per_category 4 \
       --batch_size 1000
   ```

**파라미터 설명:**
- `--als_output`: ALS 모델 출력 파일
- `--sasrec_output`: SASRec 모델 출력 파일 (선택)
- `--output_path`: 최종 앙상블 출력 파일
- `--w_als`: ALS 가중치 (기본값: 0.7)
- `--w_sasrec`: SASRec 가중치 (기본값: 0.3)
- `--max_per_category`: Top-10 내 카테고리당 최대 아이템 수 (기본값: 4)
- `--batch_size`: 배치 처리 크기 (메모리 조절, 기본값: 1000)
- `--enforce_diversity`: 카테고리 다양성 강제 여부 (기본값: True)

**메모리 부족 시 해결 방법:**
```bash
# 배치 크기 줄이기 (메모리 사용량 감소)
python ensemble_optimized.py --batch_size 500 ...

# 또는 더 작게
python ensemble_optimized.py --batch_size 250 ...
```

**예상 성능 향상:**
- 기존 앙상블 (0.1173) → 최적화 앙상블 (0.12~0.13+ 목표)
- Repeat Boost로 상위권 정확도 대폭 향상
- 카테고리 일관성으로 사용자 의도 반영

#### Basic Ensemble

ALS와 SASRec 결과를 결합하는 기본 앙상블