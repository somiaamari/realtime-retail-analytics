# Contributing

Thanks for helping improve the Real-Time Smart Retail Video Analytics Pipeline.
Contributions should follow the current roadmap and keep the foundation
cross-platform and reproducible.

## Development setup

Use Python 3.10 or 3.11 and create an isolated environment:

```shell
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS), then install the package and
development tools:

```shell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pre-commit install
```

## Before submitting

Run the quality checks from the repository root:

```shell
make lint
make format
make typecheck
make test
```

Keep changes focused, add or update tests for behavior changes, and do not
commit secrets, model weights, or video data. Use Conventional Commit messages
for changes, for example `feat(config): validate camera settings`.

## Pull requests

Describe the motivation and scope, summarize notable changes, and report the
checks you ran. Link related issues and call out any known limitations.
