@ECHO OFF
TITLE Bias Detection
CD model_scripts
CALL venv\scripts\activate
uvicorn server:app --reload --port 7001