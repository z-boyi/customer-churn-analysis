.PHONY: help install clean pipeline predict explore

help:
	@echo "Customer Churn Analysis - Available Commands"
	@echo "============================================"
	@echo "make install      - Install Python dependencies"
	@echo "make clean        - Remove cached files, models, and outputs"
	@echo "make pipeline     - Run full training & evaluation pipeline"
	@echo "make predict      - Run inference examples"
	@echo "make explore      - Launch Jupyter notebooks"
	@echo "make help         - Show this help message"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf models/*.pkl models/*.json models/*.png models/*.csv 2>/dev/null || true
	rm -rf .pytest_cache .coverage 2>/dev/null || true
	@echo "✓ Clean complete"

pipeline:
	@echo "Running training & evaluation pipeline..."
	python train_and_evaluate.py

predict:
	@echo "Running inference examples..."
	python predict.py

explore:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/

.DEFAULT_GOAL := help
