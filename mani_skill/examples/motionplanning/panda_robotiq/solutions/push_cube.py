import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.panda_robotiq.motionplanner import \
    PandaRobotiqMotionPlanningSolver

def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
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
    planner.close_gripper()

    print("robot eef rotation", env.agent.tcp.pose.sp.q)
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=np.array([0,0,1,0]))
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_RRTConnect(goal_pose)

    planner.close()
    return res
