@echo off
echo Starting Crypto Quant Liquidity Simulator...
echo.
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8501
timeout /t 3 /nobreak >nul
python -m streamlit run crypto_quant_liquidity_simulator.py

