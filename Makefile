.PHONY: install lint format typecheck test benchmark sweep clean api migrate seed openapi

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

benchmark:
	python scripts/run_all_benchmarks.py --config configs/eval.yaml

sweep:
	python scripts/sweep_tracker.py --config configs/eval.yaml --workers 1

api:
	python scripts/run_api.py

migrate:
	alembic -c alembic.ini upgrade head

seed:
	python scripts/seed_demo_data.py

openapi:
	python scripts/export_openapi.py

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist')]; [path.unlink(missing_ok=True) for path in pathlib.Path('.').glob('.coverage*')]"
