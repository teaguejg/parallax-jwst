import subprocess
import sys
import os

subprocess.Popen(
    [sys.executable, os.path.join(os.path.dirname(__file__), "examples", "example_gui.py")],
    creationflags=0x08000000,
)
