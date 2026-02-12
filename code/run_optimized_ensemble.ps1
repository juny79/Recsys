# RTX 3070 8GB 최적화 실행 스크립트
# nDCG@10 극대화를 위한 Post-Processing 앙상블

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  nDCG@10 Optimized Ensemble (RTX 3070 8GB)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 시스템 정보 출력
Write-Host "[System Info]" -ForegroundColor Yellow
Write-Host "  GPU: RTX 3070 8GB"
Write-Host "  Optimization: Memory-efficient batch processing"
Write-Host ""

# 가상환경 활성화 확인
if ($env:VIRTUAL_ENV) {
    Write-Host "[Virtual Env] ✓ Active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "[Virtual Env] Activating..." -ForegroundColor Yellow
    & ..\.venv\Scripts\Activate.ps1
}

Write-Host ""
Write-Host "[Execution Started]" -ForegroundColor Cyan
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# 최적화 앙상블 실행
python ensemble_optimized.py `
    --als_output ../output/output.csv `
    --sasrec_output ../output/output_sasrec_fixed_19.csv `
    --output_path ../output/output_optimized_final.csv `
    --max_per_category 4 `
    --batch_size 1000

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "✅ Execution Completed Successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "[Output]" -ForegroundColor Yellow
    Write-Host "  File: ../output/output_optimized_final.csv"
    
    # 파일 크기 확인
    if (Test-Path "../output/output_optimized_final.csv") {
        $fileSize = (Get-Item "../output/output_optimized_final.csv").Length / 1KB
        Write-Host "  Size: $([math]::Round($fileSize, 2)) KB"
        
        # 라인 수 확인
        $lineCount = (Get-Content "../output/output_optimized_final.csv" | Measure-Object -Line).Lines
        Write-Host "  Lines: $lineCount"
    }
} else {
    Write-Host "❌ Execution Failed (Exit Code: $exitCode)" -ForegroundColor Red
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
