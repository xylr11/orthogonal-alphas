#!/bin/bash
#SBATCH --job-name=test_backtest
#SBATCH --output=logs/test_backtest_%A_%a.out
#SBATCH --error=logs/test_backtest_%A_%a.err
#SBATCH --array=0-5   # 6 tasks total
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --mail-user=kylecm11@byu.edu 
#SBATCH --mail-type=BEGIN,END,FAIL 

DATASET="signal_data.parquet"
signals=("momentum" "meanrev" "bab")

year_starts=("1995-06-27" "1996-06-27")
year_ends=("1996-06-26" "1997-06-26")

num_signals=${#signals[@]}
num_years=${#year_starts[@]}
total_tasks=$((num_signals * num_years))

signal_index=$(( SLURM_ARRAY_TASK_ID / num_years ))
year_index=$(( SLURM_ARRAY_TASK_ID % num_years ))

signal=${signals[$signal_index]}
start=${year_starts[$year_index]}
end=${year_ends[$year_index]}

source ../.venv/bin/activate
echo "TEST RUN: signal=$signal from $start to $end"
srun python get_signal_weights.py "$DATASET" "$signal" "$start" "$end"
