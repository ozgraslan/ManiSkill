import numpy as np
import sapien
import time

from mani_skill.envs.tasks import PutCarrotOnPlateInSceneSep
from mani_skill.examples.motionplanning.panda_robotiq.motionplanner import \
    PandaRobotiqMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: PutCarrotOnPlateInSceneSep, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaRobotiqMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    carrot = env.objs[env.source_obj_name]
    plate = env.objs[env.target_obj_name]
    obb = get_actor_obb(carrot)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, carrot.pose.sp.p)
    # -------------------------------------------------------------------------- #
    # Reach Carrot
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05]) # sapien.Pose([0, 0, -0.05])
    # res = planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # planner.move_to_pose_with_screw(grasp_pose)
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Reach Plate
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(plate.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.1])
    # res = planner.move_to_pose_with_screw(goal_pose)
    planner.move_to_pose_with_RRTConnect(goal_pose)

    # -------------------------------------------------------------------------- #
    # Lower
    # -------------------------------------------------------------------------- #
    lower_pose = sapien.Pose(plate.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(lower_pose)
    res = planner.open_gripper()

    planner.close()
    return res
