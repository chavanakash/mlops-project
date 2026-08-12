"""Central place every script reads its connection info from.

Why this exists: every script in src/ (loader, training, serving) needs the
same DATABASE_URL / MLFLOW_TRACKING_URI. Hardcoding a connection string in
each file is how you end up with credentials scattered across a repo. This
module reads once from .env (never committed) so the only thing that changes
when we move to Docker/Kubernetes/cloud later is the .env file, not the code.
"""
import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
