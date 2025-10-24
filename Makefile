.PHONY: help install install-dev clean test test-unit test-integration lint format type-check security-check pre-commit run-backend run-frontend run-all docker-build docker-up docker-down logs

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  clean           Clean build artifacts and caches"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and prettier"
	@echo "  type-check      Run type checking"
	@echo "  security-check  Run security checks"
	@echo "  pre-commit      Run pre-commit hooks"
	@echo "  run-backend     Run backend development server"
	@echo "  run-frontend    Run frontend development server"
	@echo "  run-all         Run both backend and frontend"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-up       Start Docker services"
	@echo "  docker-down     Stop Docker services"
	@echo "  logs            Show Docker logs"
	@echo "  train-recipe-model Train the recipe generation model"
	@echo "  list-datasets   List available training datasets"
	@echo "  monitor-training Monitor training progress (generates samples as checkpoints are created)"
	@echo "  test-model      Test the current trained model"
	@echo "  test-checkpoint Test specific checkpoint (usage: make test-checkpoint CHECKPOINT=path)"
	@echo "  list-checkpoints List available model checkpoints"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,mlops]"
	cd frontend && npm install
	pre-commit install

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	cd frontend && npm run clean || true

# Testing
test:
	pytest

test-unit:
	pytest -m "not integration and not slow"

test-integration:
	pytest -m integration

test-ml:
	pytest -m ml

# Code Quality
lint:
	flake8 backend/ cli/
	cd frontend && npm run lint

format:
	black backend/ cli/
	isort backend/ cli/
	cd frontend && npm run format || npx prettier --write .

type-check:
	mypy backend/ cli/
	cd frontend && npm run type-check

security-check:
	bandit -r backend/ cli/
	safety check
	cd frontend && npm audit

pre-commit:
	pre-commit run --all-files

# Development servers
run-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev

run-all:
	# Note: Run in separate terminals or use a process manager like foreman
	@echo "Start backend: make run-backend"
	@echo "Start frontend: make run-frontend"

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose restart

logs:
	docker-compose logs -f

# Database operations
db-migrate:
	cd backend && alembic upgrade head

db-downgrade:
	cd backend && alembic downgrade -1

db-reset:
	cd backend && alembic downgrade base && alembic upgrade head

# Model training
train-recipe-model:
	cd cli && python train_recipe_model.py --model-output ../models/recipe_generation --epochs 50

list-datasets:
	cd cli && python train_recipe_model.py --list-datasets

# Training monitoring
monitor-training:
	cd cli && python training_monitor.py --monitor --model-dir ../models/recipe_generation

test-model:
	cd cli && python training_monitor.py --model-dir ../models/recipe_generation

test-checkpoint:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Usage: make test-checkpoint CHECKPOINT=path/to/checkpoint"; exit 1; fi
	cd cli && python training_monitor.py --test-checkpoint $(CHECKPOINT)

list-checkpoints:
	cd cli && python training_monitor.py --list-checkpoints ../models/recipe_generation

# Production deployment
deploy-staging:
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod:
	docker-compose -f docker-compose.prod.yml up -d

# Health checks
health-check:
	curl -f http://localhost:8000/health || exit 1
	curl -f http://localhost:3000 || exit 1