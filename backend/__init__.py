# Read the repo-root .env before any backend module looks at os.environ.
from backend.env import load_env_file

load_env_file()
