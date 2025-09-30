#!/bin/bash
#SBATCH --job-name=rerun_failed
#SBATCH --output=logs/rerun_failed_%A_%a.out
#SBATCH --error=logs/rerun_failed_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --mail-user=kylecm11@byu.edu
#SBATCH --mail-type=FAIL

DATA_PATH="signal_data.parquet"
SCRIPT="get_signal_weights.py"

# Define the failed runs as arrays
SIGNALS=("meanrev" "meanrev" "bab" "bab")
STARTS=("2017-06-27" "2018-06-27" "2021-06-27" "2024-06-27")
ENDS=("2018-06-26" "2019-06-26" "2022-06-26" "2025-09-15")

# Pick values based on SLURM_ARRAY_TASK_ID
SIGNAL=${SIGNALS[$SLURM_ARRAY_TASK_ID]}
START=${STARTS[$SLURM_ARRAY_TASK_ID]}
END=${ENDS[$SLURM_ARRAY_TASK_ID]}

source ../.venv/bin/activate
echo ">>> Running ${SIGNAL} from ${START} to ${END}"
python "$SCRIPT" "$DATA_PATH" "$SIGNAL" "$START" "$END" --write
