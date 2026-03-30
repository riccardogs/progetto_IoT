#!/bin/bash
#SBATCH --job-name=SleepNet_Exp_3      # Job name
#SBATCH --output=logs/%x_%j.out             # Standard output log (%x: job name, %j: job ID)
#SBATCH --error=logs/%x_%j.err              # Standard error log
#SBATCH --time=35:00:00                      # Maximum runtime (HH:MM:SS)
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=4096                   
#SBATCH --mail-type=END,FAIL                 # Notifications for job done and fail
#SBATCH --mail-user=shagupta@ethz.ch         # Email to send notifications

# ============================== #
#        Environment Setup       #
# ============================== #

# Load necessary modules (uncomment and modify if needed)
# module load python/3.8

# Activate the conda environment
echo "Activating conda environment 'sleepnet'..."
source ~/.bashrc                                # Initialize conda
conda activate sleepnet

# Verify activation
if [[ $? -ne 0 ]]; then
    echo "Failed to activate conda environment 'sleepnet'. Exiting."
    exit 1
fi
echo "Conda environment 'sleepnet' activated successfully."

# ============================== #
#         Variable Setup         #
# ============================== #

# Define project directory
PROJECT_DIR="/cluster/home/shagupta/SimpleSleepNet"

# Define experiment details
# Modify EXPERIMENT_FOLDER to Experiment_1, Experiment_2, etc.
EXPERIMENT_FOLDER="27-11-Job_3"  
# Example: "Experiment_1"

# Define configuration directory based on experiment
CONFIG_DIR="${PROJECT_DIR}/configs/${EXPERIMENT_FOLDER}"

# Define the main script path
MAIN_SCRIPT="${PROJECT_DIR}/main.py"

# Define logs directory for the experiment
EXP_LOG_DIR="${PROJECT_DIR}/logs/${EXPERIMENT_FOLDER}"

# Create logs directory if it doesn't exist
mkdir -p "${EXP_LOG_DIR}"

# ============================== #
#       Validation Checks        #
# ============================== #

# Check if configuration directory exists
if [ ! -d "${CONFIG_DIR}" ]; then
    echo "Configuration directory ${CONFIG_DIR} does not exist. Exiting."
    exit 1
fi

# Check if main.py exists
if [ ! -f "${MAIN_SCRIPT}" ]; then
    echo "Main script ${MAIN_SCRIPT} not found. Exiting."
    exit 1
fi

# ============================== #
#          Execution Loop        #
# ============================== #

echo "Starting experiments in folder: ${EXPERIMENT_FOLDER}"
echo "Configuration directory: ${CONFIG_DIR}"
echo "Logging experiments to: ${EXP_LOG_DIR}"

# Iterate over each JSON config file in the configuration directory
for config_file in "${CONFIG_DIR}"/*.json; do
    # Check if the glob actually matches files
    if [ -f "${config_file}" ]; then
        # Extract the base name of the config file (without extension)
        config_name=$(basename "${config_file}" .json)

        echo "--------------------------------------------"
        echo "Running experiment: ${config_name}"
        echo "Configuration file: ${config_file}"

        # Define the log file for this experiment
        log_file="${EXP_LOG_DIR}/${config_name}.log"

        # Execute the main.py script with the current configuration
        echo "Executing: python ${MAIN_SCRIPT} --config ${config_file}"
        python "${MAIN_SCRIPT}" --config "${config_file}" > "${log_file}" 2>&1

        # Check if the experiment was successful
        if [ $? -eq 0 ]; then
            echo "Experiment '${config_name}' completed successfully."
        else
            echo "Experiment '${config_name}' failed. Check ${log_file} for details."
        fi

        echo "--------------------------------------------"
    else
        echo "No configuration files found in ${CONFIG_DIR}."
    fi
done

echo "All experiments in '${EXPERIMENT_FOLDER}' have been executed."