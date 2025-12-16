"""
Launcher script for Crypto Quant Liquidity Simulator
Automatically opens Chrome and starts the Streamlit app
"""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def find_chrome_path():
    """Find Chrome executable on Windows"""
    possible_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    print("=" * 60)
    print("Crypto Quant Liquidity Simulator")
    print("=" * 60)
    print("\nStarting Streamlit app...")
    print("Chrome will open automatically in a few seconds.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Get the script directory
    script_dir = Path(__file__).parent
    app_file = script_dir / "crypto_quant_liquidity_simulator.py"
    
    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        input("Press Enter to exit...")
        return
    
    # Open Chrome after a short delay
    def open_browser():
        time.sleep(3)  # Wait for Streamlit to start
        url = "http://localhost:8501"
        print(f"Opening {url} in Chrome...")
        
        chrome_path = find_chrome_path()
        if chrome_path:
            subprocess.Popen([chrome_path, url])
        else:
            # Fallback to default browser
            webbrowser.open(url)
    
    # Start browser opener in background
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_file),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

