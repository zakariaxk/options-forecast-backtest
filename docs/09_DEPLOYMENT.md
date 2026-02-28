# Deployment

## Local Development

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend (serves API + UI)
uvicorn api.main:app --reload --port 8000

# Open browser
open http://127.0.0.1:8000
```

## Production
Not yet defined. MVP is local-only.
