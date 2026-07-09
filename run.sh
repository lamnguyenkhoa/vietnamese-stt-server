#!/usr/bin/env bash
cd "$(dirname "$0")"
export MODEL_DIR="${MODEL_DIR:-models}"
./venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
