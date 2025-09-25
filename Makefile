.PHONY: setup train eval lint fmt

setup:
	pip install -r requirements.txt

lint:
	black --check . && isort --check-only . && flake8 .

fmt:
	black . && isort .

# Example targets (adjust to your scripts)
train:
	python src/train.py --epochs 20 --seed 42

eval:
	python src/eval.py --checkpoint checkpoints/best.pth
