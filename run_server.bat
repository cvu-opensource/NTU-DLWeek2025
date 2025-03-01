@ECHO OFF
TITLE MODEL
CALL venv\scripts\activate
CD model_scripts
uvicorn server:app --reload