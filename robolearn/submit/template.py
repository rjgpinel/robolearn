sched_template = """#!/bin/bash
#SBATCH --job-name={job_name}

#SBATCH --qos={qos}
{queue}

#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_gpu}
#SBATCH --hint=nomultithread

#SBATCH --time={time}

# cleaning modules launched during interactive mode
module purge

# conda
. {conda_dir}/etc/profile.d/conda.sh
export LD_LIBRARY_PATH={conda_dir}/envs/bin/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY=UPDATEME
export WANDB_MODE="offline"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/lib
export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/lib
export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-8.3.1-prm3s2n7ixxt4vbajjp4z5ewfrwtuyya/include
module load cuda/11.2
cd $WORK/Code/mujoco-py/
python setup.py build
python setup.py install
"""

train_template = """conda activate {conda_env_name}

mkdir -p {log_dir}
srun --output {log_dir}/%j.out --error {log_dir}/%j.err \\
sh -c "
python -m robolearn.train \
  --task {task} \
  --log-dir {log_dir} \
  --model {model} \
  --stream-integration {stream_integration} \
  --frame-hist {frame_hist} \
  --delay-hist {delay_hist} \
  --lr {learning_rate} \
  --sched {sched} \
  --warmup-ratio {warmup_ratio} \
  --weight-decay {weight_decay} \
  --train-path {train_path} \
  --eval-path-sim {eval_path_sim} \
  --eval-path-real {eval_path_real} \
  --eval-env {eval_env} \
  --train-steps {train_steps} \
  --eval-steps {eval_steps} \
  --batch-size {batch_size} \
  --data-aug {data_aug} \
  --state {state} \
  --arch {architecture} \
  --run-name {run_name}
"
"""
