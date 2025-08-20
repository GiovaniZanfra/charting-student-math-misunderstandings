#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = charting-student-math-misunderstandings
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Test transformer setup
.PHONY: test-transformer
test-transformer: requirements
	$(PYTHON_INTERPRETER) test_transformer_setup.py





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) charting_student_math_misunderstandings/dataset.py


## Generate features (including transformer prompts)
.PHONY: features
features: requirements
	$(PYTHON_INTERPRETER) charting_student_math_misunderstandings/features.py


## Train transformer model
.PHONY: train-transformer
train-transformer: features
	$(PYTHON_INTERPRETER) charting_student_math_misunderstandings/modeling/train_transformer.py --exp $(EXP)


## Train transformer model with PEFT
.PHONY: train-transformer-peft
train-transformer-peft: features
	$(PYTHON_INTERPRETER) charting_student_math_misunderstandings/modeling/train_transformer.py --exp $(EXP) --use_peft


## Generate predictions with transformer model
.PHONY: predict-transformer
predict-transformer:
	$(PYTHON_INTERPRETER) charting_student_math_misunderstandings/modeling/predict_transformer.py --exp $(EXP)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
