@echo off
:: Use PowerShell to get a clean start timestamp
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'HH:mm:ss'"') do set STARTTIME=%%a
set "TSTART=%time%"

SETLOCAL EnableDelayedExpansion

:: --- YOUR CODE STARTS HERE ---
echo [1/7] Activating Environment...
:: Note: 'call conda' is correct for batch
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
:: --- YOUR CODE ENDS HERE ---

:: Calculate Duration using PowerShell so we don't have to deal with Batch math
for /f "tokens=*" %%a in ('powershell -command "Get-Date -Format 'HH:mm:ss'"') do set ENDTIME=%%a
for /f "tokens=*" %%a in ('powershell -command "$s=Get-Date '%STARTTIME%'; $e=Get-Date '%ENDTIME%'; $d=$e-$s; \"$($d.Minutes)m $($d.Seconds)s\""') do set DURATION=%%a

echo.
echo Started:  %STARTTIME%
echo Finished: %ENDTIME%
echo Duration: %DURATION%