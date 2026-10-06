.PHONY: install lint format typecheck test clean

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy

test:
	pytest --cov=vision_pipeline --cov-report=term-missing

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist')]; [path.unlink(missing_ok=True) for path in pathlib.Path('.').glob('.coverage*')]"
