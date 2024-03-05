VENV_NAME=venv

setup: create_venv
	@echo "Activate the venv with: \`source ./$(VENV_NAME)/bin/activate\`" ;\
	echo "Once the venv is activated, install the requirements with: \`pip install -r requirements.txt\`"

create_venv:
	python3 -m venv ./$(VENV_NAME)
	
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

reset_run:
	find . -type d -name "results" -exec rm -rf {} +
	find . -type d -name "contexts" -exec rm -rf {} +

destroy: clean reset_run
	rm -rf ./$(VENV_NAME)