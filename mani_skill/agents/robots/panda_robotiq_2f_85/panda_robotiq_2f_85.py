from copy import deepcopy
import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class PandaRobotiq(BaseAgent):
    uid = "panda_robotiq"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_robotiq_85.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0,
                    -1 / 5 * np.pi,
                    0,
                    -4 / 5 * np.pi,
                    0,
                    3 / 5 * np.pi,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]

    ee_link_name = "robotiq_hand_tcp"  # "robotiq_hand_tcp" panda_link_ee"

    arm_stiffness = 1e3
    # arm_stiffness = [
    #     1189.578752213823,
    #     1379.9426891074881,
    #     779.9513351321253,
    #     892.7242889639874,
    #     600.0,
    #     600.0,
    #     707.7313445520906,
    # ]
    arm_damping = 1e2
    arm_damping = [
        110.91641016880158,
        76.60497601124099,
        186.18323463746594,
        50.0,
        250.0,
        250.0,
        156.0774747573599,
    ]
    arm_force_limit = 100

    gripper_stiffness = 1e5
    gripper_damping = 1e3
    gripper_force_limit = 0.1
    gripper_friction = 0.05

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )
        # gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
        #     joint_names=finger_joint_names,
        #     lower=-0.1,
        #     upper=0.1,
        #     stiffness=self.gripper_stiffness,
        #     damping=self.gripper_damping,
        #     force_limit=self.gripper_force_limit,
        #     normalize_action=True,
        #     friction=self.gripper_friction,
        #     use_delta=True,
        # )

        controller_configs = dict(
            arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=dict(
                arm=arm_pd_ee_target_delta_pose_align2, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger"
        )
        self.finger1pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        )
        self.finger2pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2pad_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1pad_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger2pad_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger1pad_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def check_contact_fingers(self, actor: Actor, min_impulse=1e-6):
        assert isinstance(actor, Actor), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = sapien_utils.get_pairwise_contact_impulse(
            contacts, self.finger1pad_link, actor
        )
        rimpulse = sapien_utils.get_pairwise_contact_impulse(
            contacts, self.finger2pad_link, actor
        )

        return (
            np.linalg.norm(limpulse) >= min_impulse,
            np.linalg.norm(rimpulse) >= min_impulse,
        )

    def is_static(self, threshold: float = 0.2):
        ## this should only include robot joint vels not the gripper 
        qvel = self.robot.get_qvel()[..., :-6]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (robotiq_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)


# Tuned for the sink setup
@register_agent()
class PandaRobotiqBridgeDatasetFlatTable(PandaRobotiq):
    uid = "panda_robotiq_bridgedataset_flat_table"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
                pose=sapien.Pose(
                    [-0.00300001, -0.21, 0.39], # [-0.00300001, -0.21, 0.39], [0.1, -0.12, 0.39] [0.25, -0.12, 0.36] [0.1470000, -0.21, 0.25]
                    [0.9331967, -0.1159444, 0.255225, 0.2248576], # [0.9331967, -0.1159444, 0.255225, 0.2248576] [-0.907313, 0.0782, -0.36434, -0.194741] 
                ),
                # entity_uid="panda_link0",
                mount=self.robot.links_map["panda_link0"],
                width=640,
                # fov=1.5,
                height=480,
                near=0.01,
                far=10,
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            )
        ]
