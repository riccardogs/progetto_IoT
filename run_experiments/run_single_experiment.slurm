#!/bin/bash
#SBATCH --job-name=SleepNet_Run_%j       # Job name (%j: job ID)
#SBATCH --output=logs/%x_%j.out          # Standard output log
#SBATCH --error=logs/%x_%j.err           # Standard error log
#SBATCH --time=12:00:00                   # Maximum runtime
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --gpus=1                         # Request 1 GPU
#SBATCH --mem-per-cpu=4096                # Memory per CPU core
#SBATCH --mail-type=END,FAIL             # Notifications for job done and fail
#SBATCH --mail-user=shagupta@ethz.ch     # Email to send notifications

# ============================== #
#        Environment Setup       #
# ============================== #

# Load necessary modules (uncomment and modify if needed)
# module load python/3.8

# Activate the conda environment
echo "Activating conda environment 'sleepnet'..."
source ~/.bashrc                                # Initialize conda
conda activate sleepnet

if [[ $? -ne 0 ]]; then
    echo "Failed to activate conda environment 'sleepnet'. Exiting."
    exit 1
fi
echo "Conda environment 'sleepnet' activated successfully."

# ============================== #
#          Execution             #
# ============================== #

# Use the CONFIG_FILE passed via the --export option in sbatch
if [ -z "$CONFIG_FILE" ]; then
    echo "CONFIG_FILE is not set. Exiting."
    exit 1
fi

PROJECT_DIR="/cluster/home/shagupta/SimpleSleepNet"
MAIN_SCRIPT="${PROJECT_DIR}/main.py"

echo "Running main.py with configuration: $CONFIG_FILE"
python "${MAIN_SCRIPT}" --config "$CONFIG_FILE"

if [[ $? -eq 0 ]]; then
    echo "Experiment completed successfully."
else
    echo "Experiment failed."
    exit 1
fi
