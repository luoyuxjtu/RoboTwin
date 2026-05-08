"""Run a single RoboTwin episode with the SAPIEN viewer and stream robot data
to the terminal.

Usage:
    python run_viewer_demo.py                         # default: beat_block_hammer / demo_randomized / seed 0
    python run_viewer_demo.py beat_block_hammer demo_randomized 0

Equivalent (pipeline-wise) of:
    bash collect_data.sh beat_block_hammer demo_randomized 0
but limited to ONE episode, with the SAPIEN viewer enabled and per-step
sensor data printed to stdout instead of HDF5 saving.

What we print every ~0.1s of wall time:
  - joint angles of both arms (qpos) + normalized gripper opening
  - end-effector pose of both arms (xyz + quat)
  - contact wrench on the wrist link (force / torque in world frame, summed
    over all contacts that involve the wrist link or any gripper link,
    expressed at the wrist origin)
  - qpos of every articulated object in the scene (excluding the robots)
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs import *  # noqa: F401,F403  (registers env classes)
from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def make_env(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def load_embodiment(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_args(task_name, task_config):
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_table = yaml.load(f, Loader=yaml.FullLoader)

    def emb_file(name):
        return embodiment_table[name]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must have length 1 or 3")

    args["left_embodiment_config"] = load_embodiment(args["left_robot_file"])
    args["right_embodiment_config"] = load_embodiment(args["right_robot_file"])

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["embodiment_name"] = embodiment_name
    args["save_path"] = os.path.join(args["save_path"], task_name, task_config)

    # one-shot live demo: viewer on, no hdf5 saving, planning enabled
    args["render_freq"] = 5
    args["episode_num"] = 1
    args["use_seed"] = False
    args["save_data"] = False
    args["collect_data"] = False
    args["need_plan"] = True
    return args


def fmt(arr):
    return "[" + ", ".join(f"{x:+.4f}" for x in np.asarray(arr).reshape(-1)) + "]"


def compute_wrist_wrench(scene, wrist_link, gripper_link_names, dt):
    """Sum contact forces & torques on the wrist + gripper links,
    expressed about the wrist link origin (world frame)."""
    if wrist_link is None:
        return np.zeros(6)
    target_names = set(gripper_link_names) | {wrist_link.get_name()}
    origin = np.asarray(wrist_link.get_pose().p)
    F = np.zeros(3)
    T = np.zeros(3)
    for contact in scene.get_contacts():
        a_name = contact.bodies[0].entity.name
        b_name = contact.bodies[1].entity.name
        if a_name in target_names and b_name not in target_names:
            sign = 1.0
        elif b_name in target_names and a_name not in target_names:
            sign = -1.0
        else:
            continue
        for pt in contact.points:
            f = sign * np.asarray(pt.impulse) / dt
            r = np.asarray(pt.position) - origin
            F += f
            T += np.cross(r, f)
    return np.concatenate([F, T])


class DataLogger:
    def __init__(self, task, period_s=0.1):
        self.task = task
        self.period_s = period_s
        self.last = 0.0
        self.frame = 0
        self.t0 = time.monotonic()

        robot = task.robot
        self.left_wrist = getattr(robot.left_ee, "parent_link", None)
        self.right_wrist = getattr(robot.right_ee, "parent_link", None)
        self.left_gripper_links = [g[0].child_link.get_name() for g in robot.left_gripper if g[0] is not None]
        self.right_gripper_links = [g[0].child_link.get_name() for g in robot.right_gripper if g[0] is not None]
        self.dt = task.scene.timestep
        self.robot_entities = {robot.left_entity, robot.right_entity}

    def maybe_log(self):
        now = time.monotonic()
        if now - self.last < self.period_s:
            return
        self.last = now
        self.frame += 1

        task = self.task
        robot = task.robot

        left_q = robot.get_left_arm_real_jointState()
        right_q = robot.get_right_arm_real_jointState()
        left_ee = robot.get_left_ee_pose()
        right_ee = robot.get_right_ee_pose()
        left_wr = compute_wrist_wrench(task.scene, self.left_wrist, self.left_gripper_links, self.dt)
        right_wr = compute_wrist_wrench(task.scene, self.right_wrist, self.right_gripper_links, self.dt)

        obj_states = []
        try:
            articulations = task.scene.get_all_articulations()
        except AttributeError:
            articulations = []
        for art in articulations:
            if art in self.robot_entities:
                continue
            try:
                qpos = art.get_qpos()
            except Exception:
                continue
            if qpos is None or len(qpos) == 0:
                continue
            obj_states.append((art.get_name() or "<unnamed>", np.asarray(qpos)))

        print(f"\n[log frame {self.frame:4d} | wall t={now - self.t0:.2f}s]")
        print(f"  L qpos    : {fmt(left_q[:-1])}  gripper={left_q[-1]:+.3f}")
        print(f"  R qpos    : {fmt(right_q[:-1])}  gripper={right_q[-1]:+.3f}")
        print(f"  L ee pose : pos={fmt(left_ee[:3])}  quat={fmt(left_ee[3:])}")
        print(f"  R ee pose : pos={fmt(right_ee[:3])}  quat={fmt(right_ee[3:])}")
        print(f"  L wrench  : F={fmt(left_wr[:3])}  T={fmt(left_wr[3:])}")
        print(f"  R wrench  : F={fmt(right_wr[:3])}  T={fmt(right_wr[3:])}")
        if obj_states:
            for name, qpos in obj_states:
                print(f"  obj[{name}].qpos = {fmt(qpos)}")
        else:
            print("  (no articulated objects in scene)")


def install_logger(task, period_s=0.1):
    logger = DataLogger(task, period_s=period_s)
    orig_update = task._update_render

    def hooked_update():
        orig_update()
        logger.maybe_log()

    task._update_render = hooked_update
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", nargs="?", default="beat_block_hammer")
    parser.add_argument("task_config", nargs="?", default="demo_randomized")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log-period", type=float, default=0.1,
                        help="seconds between successive data prints")
    cli = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cli.gpu))

    task = make_env(cli.task_name)
    args = build_args(cli.task_name, cli.task_config)

    print(f"Task        : {cli.task_name}")
    print(f"Task config : {cli.task_config}")
    print(f"Seed        : {cli.seed}")
    print(f"Embodiment  : {args['embodiment_name']}")
    print(f"Render freq : every {args['render_freq']} sim steps (viewer ON)")
    print("Starting simulation. Close the SAPIEN viewer window to exit.\n")

    task.setup_demo(now_ep_num=0, seed=cli.seed, **args)
    install_logger(task, period_s=cli.log_period)

    try:
        task.play_once()
    finally:
        success = bool(getattr(task, "plan_success", False)) and bool(task.check_success())
        print("\n" + "=" * 50)
        print("Episode finished:", "SUCCESS" if success else "FAIL")
        print("=" * 50)

        viewer = getattr(task, "viewer", None)
        if viewer is not None:
            print("Holding the viewer open. Close the window to exit.")
            try:
                while not viewer.closed:
                    task._update_render()
                    viewer.render()
            except Exception:
                pass
            try:
                viewer.close()
            except Exception:
                pass

        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    main()
