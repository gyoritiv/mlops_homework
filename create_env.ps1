python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade setuptools
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
#python mlops/main.py
