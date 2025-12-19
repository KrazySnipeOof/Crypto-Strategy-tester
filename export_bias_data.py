#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export bias data from notebook to CSV for React calendar app"""
import json
import pandas as pd
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("Exporting Bias Data to CSV")
print("=" * 60)

# Read the notebook and execute cells to get bias_results
notebook_path = 'backtest.ipynb'
print(f"Reading notebook: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Execute all code cells up to the summary to get bias_results
print("Executing notebook cells to generate bias data...")
code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']

# Execute cells 0-6 (data loading through summary)
for i in range(min(7, len(code_cells))):
    cell = code_cells[i]
    source = ''.join(cell['source'])
    
    if not source.strip():
        continue
    
    if 'Old calendar code removed' in source or 'Old calendar display code removed' in source:
        continue
    
    try:
        exec(compile(source, f'<cell {i+1}>', 'exec'), globals())
    except Exception as e:
        print(f"Warning: Error in cell {i+1}: {e}")
        import traceback
        traceback.print_exc()

# Check if bias_results exists
if 'bias_results' not in globals() or bias_results is None or len(bias_results) == 0:
    print("\n❌ Error: bias_results not found or empty.")
    print("Please run the backtest notebook cells first to generate bias data.")
    sys.exit(1)

# Export to CSV
export_df = bias_results.copy()

# Convert date to string format if it's datetime
if pd.api.types.is_datetime64_any_dtype(export_df['date']):
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
else:
    export_df['date'] = pd.to_datetime(export_df['date']).dt.strftime('%Y-%m-%d')

# Ensure columns are in the right order: date, symbol, bias, scenario
export_df = export_df[['date', 'symbol', 'bias', 'scenario']]

# Create calendar-app/public directory if it doesn't exist
os.makedirs('calendar-app/public', exist_ok=True)

# Export to CSV
csv_path = 'calendar-app/public/bias-data.csv'
export_df.to_csv(csv_path, index=False)

print("\n" + "=" * 60)
print("✅ Bias Data Exported Successfully!")
print("=" * 60)
print(f"   File: {csv_path}")
print(f"   Rows: {len(export_df):,}")
print(f"   Columns: {', '.join(export_df.columns.tolist())}")
print(f"   Symbols: {', '.join(sorted(export_df['symbol'].unique()))}")
print("=" * 60)

