@echo off
REM RTX 3070 8GB 최적화 실행 스크립트 (Windows Batch)
REM nDCG@10 극대화를 위한 Post-Processing 앙상블

echo ============================================================
echo   nDCG@10 Optimized Ensemble (RTX 3070 8GB)
echo ============================================================
echo.

echo [System Info]
echo   GPU: RTX 3070 8GB
echo   Optimization: Memory-efficient batch processing
echo.

REM 가상환경 활성화
if exist "..\.venv\Scripts\activate.bat" (
    echo [Virtual Env] Activating...
    call ..\.venv\Scripts\activate.bat
) else (
    echo [Warning] Virtual environment not found
)

echo.
echo [Execution Started]
echo Time: %date% %time%
echo.

REM 최적화 앙상블 실행
python ensemble_optimized.py ^
    --als_output ../output/output.csv ^
    --sasrec_output ../output/output_sasrec_fixed_19.csv ^
    --output_path ../output/output_optimized_final.csv ^
    --max_per_category 4 ^
    --batch_size 1000

echo.
echo ============================================================
if %ERRORLEVEL% EQU 0 (
    echo Execution Completed Successfully!
    echo Output: ../output/output_optimized_final.csv
) else (
    echo Execution Failed (Exit Code: %ERRORLEVEL%^)
)
echo ============================================================
echo.

pause
