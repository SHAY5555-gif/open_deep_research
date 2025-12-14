"""Script to start LangGraph dev server."""
import subprocess
import os

os.chdir(r"c:\projects\Deep research\open_deep_research")

# Try to find uvx in common locations
uvx_paths = [
    r"C:\Users\yesha\.local\bin\uvx.exe",
    r"C:\Users\yesha\AppData\Local\Programs\uv\uvx.exe",
    "uvx",
]

for uvx_path in uvx_paths:
    try:
        print(f"Trying: {uvx_path}")
        result = subprocess.run([uvx_path, "langgraph", "dev"], check=True)
        break
    except FileNotFoundError:
        print(f"Not found: {uvx_path}")
        continue
    except Exception as e:
        print(f"Error with {uvx_path}: {e}")
        continue
