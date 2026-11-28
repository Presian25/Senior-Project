"""
Starts the dashboard
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "streamlit", "run", "App/app.py"])