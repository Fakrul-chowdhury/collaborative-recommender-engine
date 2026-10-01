# Usage: make setup && make data && make train && make evaluate
# On Windows without make, run the commands on the right-hand side directly
# (the interpreter is .venv\Scripts\python.exe).

ifeq ($(OS),Windows_NT)
PYTHON := .venv/Scripts/python.exe
else
PYTHON := .venv/bin/python
endif

.PHONY: setup data train evaluate all test clean

setup:            ## create the virtual environment and install pinned dependencies
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:             ## download Online Retail II (synthetic fallback), load SQLite, run sql/*.sql
	$(PYTHON) -m src.run_all --stage data

train:            ## tune ALS and the content fallback on the validation window
	$(PYTHON) -m src.run_all --stage tune

evaluate:         ## refit on train+validation, score the test window, write results/
	$(PYTHON) -m src.run_all --stage evaluate

all:              ## data + train + evaluate
	$(PYTHON) -m src.run_all --stage all

test:             ## run the unit tests
	$(PYTHON) -m pytest -q

clean:            ## remove the database and downloaded data
	$(PYTHON) -c "import shutil; shutil.rmtree('data', ignore_errors=True)"
