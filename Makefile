.PHONY: install test collect-universe refresh-kb daily-picks index run demo api embedding-service doctor eval-rag up desktop

install:
	python -m venv .venv
	.venv/bin/python -m pip install -e "backend[dev]"

test:
	cd backend && ../.venv/bin/python -m pytest

collect-universe:
	./stock collect-universe

refresh-kb:
	./stock refresh-kb

daily-picks:
	./stock daily-picks

index:
	./stock index

run:
	./stock

demo:
	.venv/bin/stock-agent demo

api:
	./stock api

embedding-service:
	./stock embedding-service

doctor:
	./stock doctor

eval-rag:
	./stock eval-rag

up:
	docker compose up -d

desktop:
	npm --prefix apps/desktop install
	./stock desktop
