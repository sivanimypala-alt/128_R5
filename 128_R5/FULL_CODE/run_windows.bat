@echo off
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r FULL_CODE\requirements.txt
.venv\Scripts\python.exe FULL_CODE\predict.py
.venv\Scripts\python.exe FULL_CODE\app.py
