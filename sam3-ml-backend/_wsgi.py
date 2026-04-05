"""WSGI entry point for SAM3 Label Studio ML backend."""
import os
import sys

# Ensure model.py is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from label_studio_ml.api import init_app
from model import SAM3Backend

app = init_app(model_class=SAM3Backend)

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9090)),
    )
