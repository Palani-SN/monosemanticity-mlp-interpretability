@echo off
SETLOCAL EnableDelayedExpansion

echo [1/7] Activating Environment...
call conda activate mlp || (echo Failed to activate conda & exit /b 1)

echo [2/7] Generating Dataset...
cd dataset
python data_generator.py || (echo Data generation failed & exit /b 1)
cd ..

echo [3/7] Training MLP...
python train_mlp.py || (echo MLP training failed & exit /b 1)

echo [4/7] Harvesting Activations...
python harvest_activations.py || (echo Activation harvest failed & exit /b 1)

echo [5/7] Training Sparse Autoencoder (SAE)...
python train_sae.py || (echo SAE training failed & exit /b 1)

echo [6/7] Running Feature Probe...
python feature_probe.py || (echo Feature probe failed & exit /b 1)

echo [7/7] Generating Feature Reports...
python feature_reports.py || (echo Feature report generation failed & exit /b 1)

echo.
echo ======================================================
echo Pipeline Complete: Monosemantic Features Identified.
echo ======================================================
call conda deactivate
pause