#!/bin/bash
# Starts the API in the background
uvicorn src.api.llm_apy:app --reload --host 0.0.0.0 --port 8000 &

# Starts the chainlit
chainlit run src/webapp.py --port 8001
