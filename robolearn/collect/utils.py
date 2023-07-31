import numpy as np

ROBOT_LABEL = 0
TABLE_LABEL = 1
BACKGROUND_LABEL = 2
OBJECT_LABEL = 3


def geom_name2newid(name):
    map_dict = dict(
        floor=BACKGROUND_LABEL,
        wall0=BACKGROUND_LABEL,
        wall1=BACKGROUND_LABEL,
        table=TABLE_LABEL,
        vention_tower=TABLE_LABEL,
        left_base=ROBOT_LABEL,
        left_shoulder=ROBOT_LABEL,
        left_upperarm=ROBOT_LABEL,
        left_forearm=ROBOT_LABEL,
        left_wrist1=ROBOT_LABEL,
        left_wrist2=ROBOT_LABEL,
        left_wrist3=ROBOT_LABEL,
        left_robotiq_ft300_coupling=ROBOT_LABEL,
        left_robotiq_ft300=ROBOT_LABEL,
        left_single_bracket=ROBOT_LABEL,
        left_mounted_camera=ROBOT_LABEL,
        left_gripper_body=ROBOT_LABEL,
        left_moment_arm_1=ROBOT_LABEL,
        left_truss_arm_1=ROBOT_LABEL,
        left_finger_1_tip_1=ROBOT_LABEL,
        left_finger_1_tip_2=ROBOT_LABEL,
        left_flex_finger_1=ROBOT_LABEL,
        left_moment_arm_2=ROBOT_LABEL,
        left_truss_arm_2=ROBOT_LABEL,
        left_finger_2_tip_1=ROBOT_LABEL,
        left_finger_2_tip_2=ROBOT_LABEL,
        left_flex_finger_2=ROBOT_LABEL,
        cube0=OBJECT_LABEL,
        cube1=OBJECT_LABEL,
        goal0=OBJECT_LABEL,
        goal0_left_part=OBJECT_LABEL,
        goal0_right_part=OBJECT_LABEL,
        obstacle0=OBJECT_LABEL,
        obstacle1=OBJECT_LABEL,
        bowl0_part0=OBJECT_LABEL,
        bowl0_part1=OBJECT_LABEL,
        bowl0_part2=OBJECT_LABEL,
        bowl0_part3=OBJECT_LABEL,
        bowl0_part4=OBJECT_LABEL,
        bowl0_part5=OBJECT_LABEL,
        bowl0_part6=OBJECT_LABEL,
        bowl0_part7=OBJECT_LABEL,
        bowl0_part8=OBJECT_LABEL,
        bowl0_part9=OBJECT_LABEL,
        bowl0_part10=OBJECT_LABEL,
        bowl0_part11=OBJECT_LABEL,
        bowl0_part12=OBJECT_LABEL,
        bowl0_part13=OBJECT_LABEL,
        bowl0_part14=OBJECT_LABEL,
        bowl0_part15=OBJECT_LABEL,
    )
    return map_dict[name]


def process_mask(env, mask):
    sim = env.unwrapped.scene
    for name in list(sim.model.geom_names):
        geom_id = sim.model.geom_name2id(name)
        new_geom_id = geom_name2newid(name)
        mask[mask == geom_id] = new_geom_id
    return mask


def compute_workers_seed(episodes, num_workers, initial_seed):
    seeds_worker = [(initial_seed, episodes + initial_seed)]

    if num_workers > 0:
        episodes_per_worker = episodes // num_workers
        seeds_worker = np.arange(
            initial_seed, initial_seed + episodes, episodes_per_worker
        ).tolist()
        if len(seeds_worker) == num_workers + 1:
            seeds_worker[-2] = seeds_worker[-1]
            seeds_worker = seeds_worker[:-1]
        # the last worker handles the episodes outside of the last chunk
        seeds_worker.append(initial_seed + episodes)
        # transform (i0, i1, i2, ...) in ((i0, i1), (i1, i2), ...)
        for i, _ in enumerate(seeds_worker[:-1]):
            seeds_worker[i] = (seeds_worker[i], seeds_worker[i + 1])
        seeds_worker = seeds_worker[:-1]
        assert len(seeds_worker) == num_workers

    return seeds_worker
