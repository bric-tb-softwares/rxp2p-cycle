
RESERVATION=joao.pinto_5

sbatch --reservation $RESERVATION --partition gpu-large srun_maestro.sh 1 0 0
sleep 1
sbatch --reservation $RESERVATION --partition gpu-large srun_maestro.sh 0 1 0
sleep 1
sbatch --reservation $RESERVATION --partition gpu-large srun_maestro.sh 0 1 0
sleep 1
sbatch --reservation $RESERVATION --partition gpu-large srun_maestro.sh 0 1 0
