from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resources"

def resource(filename):
    path = RESOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Resource not found: {path}")

    return str(path)