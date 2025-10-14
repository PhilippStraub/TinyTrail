# TinyLCM Raspberry Pi Zero 2W Deployment Makefile
# =================================================
# Deploy and manage drift detection scenarios on Raspberry Pi Zero 2W

# Configuration
SHELL := /bin/bash
.PHONY: help deploy deploy-quick deploy-models scenario1 scenario2 create-state1 create-state2 ssh logs sync-logs monitor clean-pi setup-pi setup-ssh list-pis test-connection test-pi

# Default configuration
PI_USER ?= pi
PI_PROJECT_PATH ?= /home/pi/tinylcm_drift_detection
RSYNC_FLAGS := -avz --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' --exclude='.vscode' --exclude='.devcontainer' --exclude='logs/*' --exclude='debug/*' --exclude='*.swp'
SSH_FLAGS := -o ConnectTimeout=5 -o StrictHostKeyChecking=no

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Help target
help:
	@echo "TinyLCM Pi Zero 2W Deployment System"
	@echo "===================================="
	@echo ""
	@echo "Setup & Configuration:"
	@echo "  make setup-ssh      - Setup SSH key authentication (run first!)"
	@echo "  make setup-pi       - Initial Pi setup (install dependencies)"
	@echo "  make list-pis       - List configured Pi devices"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy         - Deploy entire project to Pi"
	@echo "  make deploy-quick   - Deploy only Python code (faster)"
	@echo "  make deploy-models  - Deploy only model files"
	@echo ""
	@echo "Scenario Execution:"
	@echo "  make scenario1      - Run Scenario 1 (Lego drift detection)"
	@echo "  make scenario2      - Run Scenario 2 (Ball drift detection)"
	@echo "  make create-state1  - Create initial state for Scenario 1"
	@echo "  make create-state2  - Create initial state for Scenario 2"
	@echo ""
	@echo "Monitoring & Debugging:"
	@echo "  make ssh            - Open SSH session to Pi"
	@echo "  make logs           - Stream real-time logs from Pi"
	@echo "  make sync-logs      - Download logs from Pi to local"
	@echo "  make monitor        - Live monitoring of drift detection"
	@echo "  make clean-pi       - Clean logs and temp files on Pi"
	@echo "  make test-pi        - Test Pi setup and dependencies"
	@echo ""
	@echo "Options:"
	@echo "  PI=<ip>            - Specify Pi IP address"
	@echo "  PI_USER=<user>     - Override Pi username (default: pi)"
	@echo ""
	@echo "Examples:"
	@echo "  make deploy PI=192.168.1.100"
	@echo "  make scenario1 PI=192.168.1.100"

# Get or prompt for Pi IP address
define get_pi_ip
	$(eval PI_IP := $(or $(PI),$(shell \
		if [ -f .pi_hosts ] && [ -s .pi_hosts ]; then \
			echo "Available Pi devices:" >&2; \
			cat -n .pi_hosts >&2; \
			read -p "Select number or enter new IP: " choice >&2; \
			if [[ $$choice =~ ^[0-9]+$$ ]]; then \
				sed -n "$${choice}p" .pi_hosts | tr -d ' \t\n\r'; \
			else \
				echo $$choice | tr -d ' \t\n\r' | tee -a .pi_hosts; \
			fi; \
		else \
			read -p "Enter Pi IP address: " ip >&2; \
			echo $$ip | tr -d ' \t\n\r' | tee .pi_hosts; \
		fi \
	)))
	@if [ -z "$(PI_IP)" ]; then \
		echo "$(RED)Error: No Pi IP address provided$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Using Pi: $(PI_IP)$(NC)"
endef

# Setup SSH key authentication
setup-ssh:
	$(call get_pi_ip)
	@echo "$(YELLOW)Setting up SSH key authentication for $(PI_IP)...$(NC)"
	@.deployment/setup_ssh_keys.sh $(PI_IP) $(PI_USER)
	@echo "$(GREEN)SSH setup complete! You can now use passwordless commands.$(NC)"

# List configured Pis
list-pis:
	@if [ -f .pi_hosts ]; then \
		echo "Configured Pi devices:"; \
		cat -n .pi_hosts; \
	else \
		echo "No Pi devices configured yet."; \
		echo "Run 'make deploy' to add a Pi."; \
	fi

# Setup Pi with dependencies
setup-pi:
	$(call get_pi_ip)
	@echo "$(YELLOW)Setting up Pi at $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "mkdir -p $(PI_PROJECT_PATH)"
	@scp $(SSH_FLAGS) .deployment/setup_pi.sh $(PI_USER)@$(PI_IP):/tmp/
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "bash /tmp/setup_pi.sh"
	@echo "$(GREEN)Pi setup complete!$(NC)"

# Deploy entire project
deploy:
	$(call get_pi_ip)
	@echo "$(YELLOW)Deploying to $(PI_IP)...$(NC)"
	@echo "Creating project structure..."
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "mkdir -p $(PI_PROJECT_PATH)"
	@echo "Syncing files..."
	@rsync $(RSYNC_FLAGS) \
		--exclude='examples/data/*' \
		--exclude='examples/model_training/*' \
		./ $(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/
	@echo "Setting permissions..."
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "find $(PI_PROJECT_PATH) -name '*.py' -exec chmod +x {} \;"
	@echo "$(GREEN)Deployment complete!$(NC)"

# Quick deploy (Python code only)
deploy-quick:
	$(call get_pi_ip)
	@echo "$(YELLOW)Quick deploying Python code to $(PI_IP)...$(NC)"
	@rsync $(RSYNC_FLAGS) \
		--include='*/' \
		--include='*.py' \
		--exclude='*' \
		tinylcm_python/ $(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/tinylcm_python/
	@rsync $(RSYNC_FLAGS) \
		--include='*/' \
		--include='*.py' \
		--exclude='*' \
		examples/python_examples/ $(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/examples/python_examples/
	@echo "$(GREEN)Quick deployment complete!$(NC)"

# Deploy models only
deploy-models:
	$(call get_pi_ip)
	@echo "$(YELLOW)Deploying models to $(PI_IP)...$(NC)"
	@rsync $(RSYNC_FLAGS) \
		examples/python_examples/scenario_lego_drift/model/ \
		$(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/examples/python_examples/scenario_lego_drift/model/
	@if [ -d "examples/python_examples/scenario_ball_drift/model/" ]; then \
		rsync $(RSYNC_FLAGS) \
			examples/python_examples/scenario_ball_drift/model/ \
			$(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift/model/; \
	fi
	@echo "$(GREEN)Model deployment complete!$(NC)"

# Create initial state for Scenario 1
create-state1:
	$(call get_pi_ip)
	@echo "$(YELLOW)Creating initial state for Scenario 1 on $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
		"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
		 cd $(PI_PROJECT_PATH)/examples/python_examples/scenario_lego_drift && \
		 python3 create_initial_state.py"
	@echo "$(GREEN)Initial state created!$(NC)"

# Create initial state for Scenario 2
create-state2:
	$(call get_pi_ip)
	@echo "$(YELLOW)Creating initial state for Scenario 2 on $(PI_IP)...$(NC)"
	@if ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "[ -d $(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift ]"; then \
		ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
			"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
			 cd $(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift && \
			 python3 create_initial_state.py"; \
		echo "$(GREEN)Initial state created!$(NC)"; \
	else \
		echo "$(YELLOW)Scenario 2 not found. Deploy first with: make deploy$(NC)"; \
	fi

# Run Scenario 1 (Lego drift detection)
scenario1:
	$(call get_pi_ip)
	@echo "$(YELLOW)Running Scenario 1 (Lego Drift) on $(PI_IP)...$(NC)"
	@echo "Press Ctrl+C to stop"
	@ssh $(SSH_FLAGS) -t $(PI_USER)@$(PI_IP) \
		"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
		 cd $(PI_PROJECT_PATH)/examples/python_examples/scenario_lego_drift && \
		 python3 main.py --config config.json"

# Run Scenario 2 (Ball drift detection)
scenario2:
	$(call get_pi_ip)
	@echo "$(YELLOW)Running Scenario 2 (Ball Drift) on $(PI_IP)...$(NC)"
	@if ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "[ -f $(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift/main.py ]"; then \
		echo "Press Ctrl+C to stop"; \
		ssh $(SSH_FLAGS) -t $(PI_USER)@$(PI_IP) \
			"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
			 cd $(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift && \
			 python3 main.py --config config.json"; \
	else \
		echo "$(YELLOW)Scenario 2 not found. Deploy first with: make deploy$(NC)"; \
	fi

# SSH into Pi
ssh:
	$(call get_pi_ip)
	@echo "$(GREEN)Connecting to Pi at $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) -t $(PI_USER)@$(PI_IP)

# Stream logs from Pi
logs:
	$(call get_pi_ip)
	@echo "$(YELLOW)Streaming logs from $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
		"tail -f $(PI_PROJECT_PATH)/examples/python_examples/scenario_*/logs/*.log 2>/dev/null || echo 'No logs found. Run a scenario first.'"

# Sync logs from Pi to local
sync-logs:
	$(call get_pi_ip)
	@echo "$(YELLOW)Syncing logs from $(PI_IP)...$(NC)"
	@mkdir -p ./pi_logs/$(PI_IP)
	@rsync -avz $(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/examples/python_examples/scenario_lego_drift/logs/ \
		./pi_logs/$(PI_IP)/scenario1/ 2>/dev/null || true
	@rsync -avz $(PI_USER)@$(PI_IP):$(PI_PROJECT_PATH)/examples/python_examples/scenario_ball_drift/logs/ \
		./pi_logs/$(PI_IP)/scenario2/ 2>/dev/null || true
	@echo "$(GREEN)Logs synced to ./pi_logs/$(PI_IP)/$(NC)"

# Live monitoring
monitor:
	$(call get_pi_ip)
	@echo "$(YELLOW)Starting drift monitor on $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
		"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
		 cd $(PI_PROJECT_PATH) && python3 .deployment/monitor.py"

# Clean Pi logs and temp files
clean-pi:
	$(call get_pi_ip)
	@echo "$(YELLOW)Cleaning Pi at $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
		"find $(PI_PROJECT_PATH) -name '*.pyc' -delete; \
		 find $(PI_PROJECT_PATH) -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true; \
		 rm -rf $(PI_PROJECT_PATH)/examples/python_examples/scenario_*/logs/* 2>/dev/null || true; \
		 rm -rf $(PI_PROJECT_PATH)/examples/python_examples/scenario_*/debug/* 2>/dev/null || true"
	@echo "$(GREEN)Pi cleaned!$(NC)"

# Test SSH connection
test-connection:
	$(call get_pi_ip)
	@echo "$(YELLOW)Testing connection to $(PI_IP)...$(NC)"
	@if ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) "echo 'Connection successful'" 2>/dev/null; then \
		echo "$(GREEN)✓ Connection successful$(NC)"; \
	else \
		echo "$(RED)✗ Connection failed$(NC)"; \
		exit 1; \
	fi

# Test Pi setup
test-pi:
	$(call get_pi_ip)
	@echo "$(YELLOW)Testing Pi setup at $(PI_IP)...$(NC)"
	@ssh $(SSH_FLAGS) $(PI_USER)@$(PI_IP) \
		"export PYTHONPATH=$(PI_PROJECT_PATH):\$$PYTHONPATH && \
		 cd $(PI_PROJECT_PATH) && python3 .deployment/test_setup.py"

# Default target
.DEFAULT_GOAL := help