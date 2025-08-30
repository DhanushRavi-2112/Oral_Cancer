@echo off
echo ========================================
echo Running Full Model Evaluation
echo ========================================
echo.

echo Step 1: Testing individual predictions
python test_both_models.py

echo.
echo Step 2: Running comprehensive comparison
python src/evaluation/compare_models.py --data_dir data --device cpu --save_dir outputs

echo.
echo Step 3: Checking results
if exist outputs\evaluation_report.txt (
    echo.
    echo === EVALUATION SUMMARY ===
    type outputs\evaluation_report.txt | findstr /C:"RECOMMENDATIONS"
    echo.
    echo Full report saved at: outputs\evaluation_report.txt
) else (
    echo No evaluation report generated. Check for errors above.
)

echo.
echo ========================================
echo Evaluation Complete!
echo ========================================
pause