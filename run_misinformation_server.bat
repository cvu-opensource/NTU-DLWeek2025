@ECHO OFF
TITLE Misinformation Detection
CD misinformation_scripts
CALL venv\scripts\activate
uvicorn misinformation_pipeline:app --reload --port 7001