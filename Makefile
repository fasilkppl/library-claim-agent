.PHONY: setup models run phone check test test-slow lint types eval reprice marker submission ar-check ar-sweep

setup:            ## install Python deps (uv) and browser for the E2E test
	uv sync
	uv run playwright install chromium

models:           ## download all local models (~8 GB)
	./scripts/fetch_models.sh

run:              ## start the agent on http://localhost:8000
	uv run claim-agent

phone:            ## forward the phone (USB debugging on) to the Mac: open http://localhost:8000 in Chrome on the phone
	adb reverse tcp:8000 tcp:8000

check: lint types test

lint:
	uv run ruff check src tests --no-fix
	uv run ruff format --check src tests

types:
	uv run mypy

test:             ## fast unit tests (no models, no network)
	uv run pytest -q

test-slow:        ## replay pipeline with real models + browser E2E (server must be running for the latter)
	uv run pytest -m slow -s -o addopts=""

eval:             ## make eval SWEEP=<id> TRUTH=ground_truth/mine
	uv run python -m claimagent.pipeline eval --sweep $(SWEEP) --truth $(TRUTH)

submission:       ## make submission SWEEP=<id> TRUTH=ground_truth/mine: packet + frames zip, results sheet, PDFs
	uv run python scripts/package_submission.py --sweep $(SWEEP) --truth $(TRUTH)

reprice:          ## make reprice SWEEP=<id> COUNTRY=US
	uv run python -m claimagent.pipeline reprice --sweep $(SWEEP) --country $(COUNTRY) -n 10

ar-check:         ## make ar-check LABEL=native SECONDS=45: record ARCore's log while an AR app runs (phone on USB)
	uv run python scripts/ar_check.py record $(LABEL) $(SECONDS)

ar-sweep:         ## make ar-sweep SWEEP=<id>: AR tracking segments, trust and depth of a sweep
	uv run python scripts/ar_check.py sweep $(SWEEP)

marker:
	uv run python scripts/make_marker.py
