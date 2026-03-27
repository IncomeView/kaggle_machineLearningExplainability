import subprocess
from pathlib import Path

def get_git_version():
    try:
        tag = subprocess.check_output(["git", "describe", "--tags"]).decode().strip()
        return tag
    except:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        return f"0.0.0+{sha}"

version = get_git_version()
Path("VERSION").write_text(version)

print("Versão gerada:", version)
