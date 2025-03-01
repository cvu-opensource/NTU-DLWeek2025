@ECHO OFF
TITLE CLASSIFIER
CALL venv\scripts\activate
CD model_scripts
uvicorn server:app --reload --port 7001