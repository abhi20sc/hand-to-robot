"""Helpers for the LIBERO task 'pick up the alphabet soup (a can) and place it in the basket'."""
import glob
import os

import numpy as np

CAN = "alphabet_soup_1"
BASKET = "basket_1"
OTHER_OBJECTS = ["salad_dressing_1", "cream_cheese_1", "milk_1", "tomato_sauce_1", "butter_1"]

# Heights in metres, found by probing the sim with a scripted grasp
GRASP_DZ = 0.027   # gripper this far above the can's origin closes around the can
HOVER_DZ = 0.15    # line up this far above the can before going down
CARRY_Z = 0.25     # travel height: clears the basket rim and the other objects
PLACE_DZ = 0.15    # open the gripper this far above the basket's origin

# Region the can is placed in when generating training data (metres, world frame)
CAN_X_RANGE = (-0.22, 0.10)
CAN_Y_RANGE = (-0.30, 0.05)


def bddl_path():
    from libero.libero import get_libero_path
    folder = os.path.join(get_libero_path("bddl_files"), "libero_object")
    return [f for f in glob.glob(os.path.join(folder, "*.bddl")) if "alphabet_soup" in f][0]


def make_env(render=False, size=256):
    """render=True gives camera images (for GIFs); render=False is physics only and faster."""
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv
    if render:
        return OffScreenRenderEnv(bddl_file_name=bddl_path(), camera_heights=size, camera_widths=size, horizon=2000)
    return ControlEnv(bddl_file_name=bddl_path(), use_camera_obs=False,
                      has_offscreen_renderer=False, has_renderer=False, horizon=2000)


def settle(env, steps=10):
    """Do nothing for a few steps so objects come to rest. Returns the latest observation."""
    a = np.zeros(7)
    a[6] = -1.0  # gripper open
    for _ in range(steps):
        obs, _, _, _ = env.step(a)
    return obs


def spot_is_free(obs, xy):
    """False if xy is on top of another object or the basket."""
    for name in OTHER_OBJECTS + [BASKET]:
        min_dist = 0.13 if name == BASKET else 0.075
        if np.linalg.norm(obs[f"{name}_pos"][:2] - xy) < min_dist:
            return False
    return True


def set_can_xy(env, xy):
    """Move the can (upright) to xy on the table and let it settle."""
    sim = env.env.sim
    joint = env.env.objects_dict[CAN].joints[0]
    q = sim.data.get_joint_qpos(joint).copy()
    q[:3] = [xy[0], xy[1], 0.04]
    q[3:] = [0.0, 0.0, 0.70710678, 0.70710678]  # upright, same as LIBERO's default
    sim.data.set_joint_qpos(joint, q)
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()
    return settle(env)


def sample_can_xy(rng, obs, x_range=CAN_X_RANGE, y_range=CAN_Y_RANGE):
    """Random free spot inside the given region (None if nothing free was found)."""
    for _ in range(200):
        xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
        if spot_is_free(obs, xy):
            return xy
    return None


def lowdim(obs):
    """The 11 numbers the policy sees: gripper position (3), finger opening (2),
    can position (3), basket position (3)."""
    return np.concatenate([obs["robot0_eef_pos"], obs["robot0_gripper_qpos"],
                           obs[f"{CAN}_pos"], obs[f"{BASKET}_pos"]]).astype(np.float32)


def success(env):
    return bool(env.env._check_success())
