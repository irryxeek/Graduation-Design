PYTHON ?= python3
DATA_DIR ?= Data/Processed_ATP_WAP_2025
MODEL_PATH ?= experiments/atp_wap_2025_hw4_hmon_g005/enhanced_ro_diffusion_best.pth
DDIM_STEPS ?= 50

.PHONY: help test train-paper eval-paper eval-paper-physical app

help:
	@echo "Available targets:"
	@echo "  make test                 - run lightweight unit tests"
	@echo "  make train-paper          - run the paper-mainline training"
	@echo "  make eval-paper           - evaluate the paper-mainline model in standardized space"
	@echo "  make eval-paper-physical  - evaluate the paper-mainline model in physical space"
	@echo "  make app                  - launch the Streamlit demo"

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

train-paper:
	$(PYTHON) src/train.py \
		--mode multi \
		--model enhanced \
		--data_dir $(DATA_DIR) \
		--epochs 50 \
		--batch_size 64 \
		--patience 15 \
		--var_weights 1,1,4 \
		--monitor_target humidity \
		--humidity_grad_weight 0.05

eval-paper:
	$(PYTHON) src/evaluate.py \
		--model_path $(MODEL_PATH) \
		--model_type enhanced \
		--data_dir $(DATA_DIR) \
		--out_channels 3 \
		--sampler ddim \
		--ddim_steps $(DDIM_STEPS) \
		--n_samples 0 \
		--batch_size 64 \
		--metric_space standardized

eval-paper-physical:
	$(PYTHON) src/evaluate.py \
		--model_path $(MODEL_PATH) \
		--model_type enhanced \
		--data_dir $(DATA_DIR) \
		--out_channels 3 \
		--sampler ddim \
		--ddim_steps $(DDIM_STEPS) \
		--n_samples 0 \
		--batch_size 64 \
		--metric_space physical

app:
	streamlit run ro_retrieval/app/streamlit_app.py
