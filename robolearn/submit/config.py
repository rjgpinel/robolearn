"""
Scheduler cfg
"""

sched_cfg = dict(
    qos="qos_gpu-t3",
    time="20:00:00",
    nodes=1,
    gpus_per_node=1,
    cpus_per_gpu=10,
    conda_env_name="muse",
    conda_dir="$WORK/miniconda3/",
    queue="",
)

dev_cfg = dict(qos="qos_gpu-dev", time="02:00:00")

"""
Base cfg
"""

base_cfg = dict(
    log_dir="{checkpoint_dir}/{exp_name}",
    run_name="{exp_name}",
    frame_hist="1,2",
    delay_hist=0,
    model="resnet18",
    stream_integration="late",
    learning_rate=3e-4,
    warmup_ratio=0.02,
    weight_decay=0.0,
    sched="cosine",
    batch_size=32,
    data_aug="iros23_s2r",
    state="gripper_pose_current",
)


"""
Task cfgs
"""

task_cfgs = dict(
    pose=dict(
        checkpoint_dir="$WORK/models_pose",
        eval_path_sim="''",
        eval_path_real="''",
        train_steps=400000,
        eval_steps=25000,
        eval_env=None,
    ),
    bc=dict(
        checkpoint_dir="$WORK/models_bc",
        eval_path_sim="''",
        eval_path_real="''",
        train_steps=400000,
        eval_steps=100000,
    ),
)
