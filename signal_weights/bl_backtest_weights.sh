#!/bin/bash
#SBATCH --job-name=bl_backtest
#SBATCH --output=logs/bl_backtest_%A_%a.out
#SBATCH --error=logs/bl_backtest_%A_%a.err
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --mail-user=kylecm11@byu.edu 
#SBATCH --mail-type=BEGIN,END,FAIL 

DATASET="bl_returns_predbeta.parquet"

# Year windows: [start, end] with no overlap
year_starts=(
  "1995-06-27" "1996-06-27" "1997-06-27" "1998-06-27" "1999-06-27"
  "2000-06-27" "2001-06-27" "2002-06-27" "2003-06-27" "2004-06-27"
  "2005-06-27" "2006-06-27" "2007-06-27" "2008-06-27" "2009-06-27"
  "2010-06-27" "2011-06-27" "2012-06-27" "2013-06-27" "2014-06-27"
  "2015-06-27" "2016-06-27" "2017-06-27" "2018-06-27" "2019-06-27"
  "2020-06-27" "2021-06-27" "2022-06-27" "2023-06-27" "2024-06-27"
)

year_ends=(
  "1996-06-26" "1997-06-26" "1998-06-26" "1999-06-26" "2000-06-26"
  "2001-06-26" "2002-06-26" "2003-06-26" "2004-06-26" "2005-06-26"
  "2006-06-26" "2007-06-26" "2008-06-26" "2009-06-26" "2010-06-26"
  "2011-06-26" "2012-06-26" "2013-06-26" "2014-06-26" "2015-06-26"
  "2016-06-26" "2017-06-26" "2018-06-26" "2019-06-26" "2020-06-26"
  "2021-06-26" "2022-06-26" "2023-06-26" "2024-06-26" "2025-09-15"
)

num_years=${#year_starts[@]}
total_tasks=$((num_years))

if [ $SLURM_ARRAY_TASK_ID -ge $total_tasks ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (max $((total_tasks-1)))."
  exit 1
fi

year_index=$((SLURM_ARRAY_TASK_ID))

start=${year_starts[$year_index]}
end=${year_ends[$year_index]}

source ../.venv/bin/activate
echo "Running bl from $start to $end"
srun python get_bl_weights.py "$DATASET" "$start" "$end" --write