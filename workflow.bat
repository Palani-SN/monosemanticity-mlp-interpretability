@echo off
SETLOCAL EnableDelayedExpansion

echo [1/6] Activating Environment...
call conda activate mlp || (echo Failed to activate conda & exit /b 1)

echo [2/6] Generating Dataset...
cd dataset
python data_generator.py || (echo Data generation failed & exit /b 1)
cd ..

echo [3/6] Training MLP...
python train_mlp.py || (echo MLP training failed & exit /b 1)

echo [4/6] Harvesting Activations...
python harvest_activations.py || (echo Activation harvest failed & exit /b 1)

echo [5/6] Training Sparse Autoencoder (SAE)...
python train_sae.py || (echo SAE training failed & exit /b 1)

echo [6/6] Running Feature Probe...
python feature_probe.py || (echo Feature probe failed & exit /b 1)

echo.
echo ======================================================
echo Pipeline Complete: Monosemantic Features Identified.
echo ======================================================
call conda deactivate
pause