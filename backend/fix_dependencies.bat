@echo off
echo Fixing Python dependencies...

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing compatible spaCy version...
pip install --upgrade "spacy>=3.7.4,<3.8.0"

echo Installing compatible pydantic...
pip install --upgrade "pydantic>=2.5.0,<3.0.0"

echo Installing other requirements...
pip install -r requirements.txt

echo Dependencies fixed! You can now run:
echo uvicorn app.main:app --reload

pause