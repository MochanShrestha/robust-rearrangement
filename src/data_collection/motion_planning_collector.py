"""
Motion planning data collector for peg-in-hole task.
Implements algorithmic approach instead of teleoperation.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import scipy.spatial.transform as st
from enum import Enum

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from furniture_bench.envs.observation import FULL_OBS
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

import torch

from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import pickle_data
from src.data_collection.collect_enum import CollectEnum

from isaacgym import gymapi, gymutil


class PlanningState(Enum):
    """States for the motion planning state machine."""
    INITIAL_PAUSE = -1  # New state
    ANALYZE_SCENE = 0
    MOVE_TO_PEG_APPROACH = 1
    MOVE_TO_PEG = 2
    GRASP_PEG = 3
    LIFT_PEG = 4
    MOVE_TO_HOLE_APPROACH = 5
    INSERT_PEG = 6
    RELEASE_PEG = 7
    RETREAT = 8
    COMPLETE = 9


class MotionPlanningCollector:
    """
    Data collector that uses motion planning for peg-in-hole assembly.
    """
    
    def __init__(
        self,
        data_path: str,
        furniture: str = "factory_peg_hole",
        randomness: Randomness = Randomness.LOW,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        headless: bool = False,
        num_demos: int = 100,
        save_failure: bool = True,
        ctrl_mode: str = "diffik",
        compress_pickles: bool = True,
        verbose: bool = True,
    ):
        """Initialize the motion planning collector."""
        
        self.env = FurnitureRLSimEnv(
            furniture=furniture,
            obs_keys=FULL_OBS,
            headless=headless,
            max_env_steps=1000,
            num_envs=1,
            act_rot_repr="quat",
            action_type="delta",
            manual_done=False,
            resize_img=True,
            np_step_out=False,
            channel_first=False,
            randomness=randomness,
            compute_device_id=compute_device_id,
            graphics_device_id=graphics_device_id,
            ctrl_mode=ctrl_mode,
        )
        
        self.data_path = Path(data_path)
        self.furniture = furniture
        self.num_demos = num_demos
        self.save_failure = save_failure
        self.compress_pickles = compress_pickles
        self.verbose = verbose
        
        # Planning parameters
        self.approach_height = 0.05  # Height above target for approach
        self.insertion_depth = 0.035  # How deep to insert peg
        self.move_speed = 0.1  # Maximum movement per step
        self.rotation_speed = 0.1  # Maximum rotation per step
        self.grasp_threshold = 0.01  # Distance threshold for grasping
        self.pos_tolerance = 0.005  # Position tolerance for waypoints
        self.ori_tolerance = 0.1  # Orientation tolerance (radians)
        
        # State tracking
        self.current_state = PlanningState.INITIAL_PAUSE # Updated initial state
        self.target_waypoint = None
        self.target_orientation = None
        self.peg_pose = None
        self.hole_pose = None
        self.gripper_closed = False
        self.grasp_step_counter = 0  # Counter for grasp duration
        self.pause_step_counter = 0 # Counter for initial pause
        
        # Data collection
        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0
        self._reset_collector_buffer()
    
    def _get_global_ee_pos(self) -> np.ndarray:
        """Helper function to get the current end-effector position as a squeezed NumPy array.
        The end-effector position is in the global frame of the environment."""
        pos_tensor = self.env.rb_states[self.env.ee_idxs[0], :3]
        pos_numpy_squeezed = pos_tensor.cpu().numpy().squeeze()
        return pos_numpy_squeezed

    def verbose_print(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[MotionPlanning] {msg}")
    
    def collect(self):
        """Main collection loop."""
        self.verbose_print("Starting motion planning data collection!")
        
        while self.num_success < self.num_demos:
            try:
                success = self._collect_single_trajectory()
                if success:
                    self.num_success += 1
                else:
                    self.num_fail += 1
                    
                self.traj_counter += 1
                self.verbose_print(f"Completed trajectory {self.traj_counter}: "
                                 f"Success: {self.num_success}, Fail: {self.num_fail}")
                
            except Exception as e:
                self.verbose_print(f"Error in trajectory collection: {e}")
                self.num_fail += 1
                self.traj_counter += 1
        
        self.verbose_print(f"Collection complete! {self.num_success}/{self.traj_counter} successful.")
    
    def _collect_single_trajectory(self) -> bool:
        """Collect a single trajectory using motion planning."""
        # Reset environment and state
        obs = self.env.reset()
        self._reset_collector_buffer() # Resets grasp_step_counter and pause_step_counter
        self.current_state = PlanningState.INITIAL_PAUSE # Updated initial state
        self.gripper_closed = False
        self.grasp_step_counter = 0 # Explicitly reset here too for clarity
        self.pause_step_counter = 0 # Explicitly reset here for clarity
        
        done = False
        step_count = 0
        max_steps = 500
        
        while step_count < max_steps:
            try:
                # Plan next action based on current state
                action = self._plan_next_action(obs)
                
                if action is None:
                    self.verbose_print("Planning failed - no valid action")
                    return False
                
                # Execute action
                next_obs, reward, done, info = self.env.step(action)

                # Store transition
                self._store_transition(obs, action, reward)

                if done.item():
                    self.verbose_print("Environment done state reached.")
                
                # Check if task is complete
                if self.current_state == PlanningState.COMPLETE:
                    success = done.item()
                    self._save_trajectory(success)
                    return success
                
                obs = next_obs
                step_count += 1
                
            except Exception as e:
                self.verbose_print(f"Error in step {step_count}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Timeout or other failure
        self.verbose_print(f"Trajectory timeout after {step_count} steps")
        self._save_trajectory(False)
        return False
    
    def _plan_next_action(self, obs) -> Optional[torch.Tensor]:
        """Plan the next action based on current state and observations."""
        
        if self.current_state == PlanningState.INITIAL_PAUSE:
            return self._initial_pause_action()

        elif self.current_state == PlanningState.ANALYZE_SCENE:
            return self._analyze_scene(obs)
            
        elif self.current_state == PlanningState.MOVE_TO_PEG_APPROACH:
            return self._move_to_waypoint(obs, self.target_waypoint, self.target_orientation)
            
        elif self.current_state == PlanningState.MOVE_TO_PEG:
            return self._move_to_waypoint(obs, self.peg_pose[:3] + [0, 0, 0.03], self.peg_pose[3:7])
            
        elif self.current_state == PlanningState.GRASP_PEG:
            return self._grasp_action(obs)
            
        elif self.current_state == PlanningState.LIFT_PEG:
            return self._lift_action(obs)
            
        elif self.current_state == PlanningState.MOVE_TO_HOLE_APPROACH:
            return self._move_to_waypoint(obs, self.target_waypoint, self.target_orientation)
            
        elif self.current_state == PlanningState.INSERT_PEG:
            return self._insert_action(obs)
            
        elif self.current_state == PlanningState.RELEASE_PEG:
            return self._release_action(obs)
            
        elif self.current_state == PlanningState.RETREAT:
            return self._retreat_action(obs)
        
        return None
    
    def _initial_pause_action(self) -> torch.Tensor:
        """Perform no action for a few steps at the beginning."""
        self.pause_step_counter += 1
        self.max_steps = 3  # Define how many steps to pause initially
        self.verbose_print(f"Initial pause: step {self.pause_step_counter}/{self.max_steps}")

        if self.pause_step_counter >= self.max_steps:
            self.current_state = PlanningState.ANALYZE_SCENE
            self.pause_step_counter = 0 # Reset for potential future use if needed
            self.verbose_print("Initial pause complete, moving to ANALYZE_SCENE state.")
            # Immediately call analyze_scene to avoid a skipped step
            # This assumes _analyze_scene also returns an action
            # obs is not available here, so we must rely on the next call to _plan_next_action
            # For now, return a zero action and let the next cycle handle ANALYZE_SCENE
        
        # Return a zero action (no movement, gripper open)
        action = torch.zeros(8).to(self.env.device)
        action[6] = 1.0
        action[7] = -1.0  # Keep gripper open
        return action.unsqueeze(0)

    def _analyze_scene(self, obs) -> torch.Tensor:
        """Analyze the scene to find peg and hole positions."""
        # try:
        parts_poses = obs["parts_poses"].cpu().numpy()
        parts_poses_sim = obs["parts_poses_sim"].cpu().numpy()
        
        # Debug: print the structure of parts_poses
        self.verbose_print(f"Parts poses shape: {parts_poses.shape}")
        self.verbose_print(f"Parts poses content: {parts_poses}")
        
        # self.peg_pose = parts_poses[0, 0:7]
        self.peg_pose = parts_poses_sim[0, 0:7]  # Use simulated pose for peg
        # self.hole_pose = parts_poses[0, 7:14]
        self.hole_pose = parts_poses_sim[0, 7:14]  # Use simulated pose for hole
        
        # Set approach waypoint above peg
        self.target_waypoint = self.peg_pose[:3].copy()
        self.target_waypoint[2] += self.approach_height  # Move up
        self.target_orientation = self.peg_pose[3:7]  # Same orientation as peg
        
        self.verbose_print(f"Found peg at: {self.peg_pose[:3]}")
        self.verbose_print(f"Found hole at: {self.hole_pose[:3]}")
        
        self.current_state = PlanningState.MOVE_TO_PEG_APPROACH
        return self._move_to_waypoint(obs, self.target_waypoint, self.target_orientation)
        
    def _move_to_waypoint(self, obs, target_pos: np.ndarray, target_quat: np.ndarray) -> torch.Tensor:
        """Generate action to move to a target waypoint."""
        current_pos_local_to_base, current_quat_tensor = self.env.get_ee_pose() # EE pose relative to robot base
        current_pos = current_pos_local_to_base.cpu().numpy().squeeze()
        current_quat = current_quat_tensor.cpu().numpy().squeeze()

        # Print he current and target orientation as debug information in degress along x,y and z
        current_rot = st.Rotation.from_quat(current_quat)
        # Original target orientation from input (e.g., peg or hole orientation)
        original_target_rot = st.Rotation.from_quat(target_quat)
        
        # Get Euler angles of the original target orientation
        original_target_euler_xyz = original_target_rot.as_euler('xyz', degrees=True)
        
        # Define desired robot orientation: 180 deg around X, 0 deg around Y, original Z rotation
        # The z-axis rotation of the original target_rot should be preserved for the robot's tool.
        # We assume the robot's base frame aligns with the world frame for Euler angle interpretation.
        # The gripper should point downwards, so a 180-degree rotation around the world X-axis is typical.
        # If the environment's X-axis is forward, then this makes the gripper point down.
        # If the environment's Z-axis is up, then a 180-degree rotation around X makes the tool point along -Z.
        # We want the tool's Z-axis to point downwards (e.g. towards the table).
        # A standard robot end-effector orientation for picking/placing often involves
        # the robot's Z-axis (tool axis) pointing downwards.
        # If world +Z is up, then robot +X forward, +Y left, +Z up (standard).
        # To make tool point down (along world -Z), we need robot's +Z axis to align with world -Z.
        # This is often achieved by a 180-degree rotation around the robot's X-axis or Y-axis,
        # depending on the default alignment of the tool.
        # Let's assume a 180-degree rotation around the world X-axis for the EE,
        # and preserve the original target's Z-axis rotation (yaw).
        
        # Desired Euler angles for the robot's end-effector:
        # X-axis rotation: 180 degrees (to point downwards, assuming standard coordinate frames)
        # Y-axis rotation: 0 degrees
        # Z-axis rotation: Use the Z-rotation from the original target_quat (peg/hole orientation)
        desired_euler_xyz = np.array([180, 0, original_target_euler_xyz[2]])
        
        # Create the desired robot rotation object
        desired_robot_rot = st.Rotation.from_euler('xyz', desired_euler_xyz, degrees=True)
        
        # Convert the desired robot rotation to quaternion
        target_quat_robot = desired_robot_rot.as_quat()

        current_euler = current_rot.as_euler('xyz', degrees=True)
        # Target orientation for the robot, not the object
        target_robot_euler = desired_robot_rot.as_euler('xyz', degrees=True) 
        self.verbose_print(f"Current orientation (deg): {current_euler} in quaternion: {current_quat}")
        self.verbose_print(f"Desired Robot EE orientation (deg): {target_robot_euler} in quaternion: {target_quat_robot}")
        self.verbose_print(f"Original Target Object orientation (deg): {original_target_rot.as_euler('xyz', degrees=True)} in quaternion: {target_quat}")


        # --- Visualization: Draw marker and line in Isaac Gym ---
        gym = self.env.isaac_gym
        viewer = self.env.viewer

                           # RGB arrows
        box_geom = gymutil.WireframeBoxGeometry(0.05, 0.05, 0.05,             # wire box
                                            gymapi.Transform(),
                                            color=(1, 0, 0))
        box_geom_target = gymutil.WireframeBoxGeometry(0.05, 0.05, 0.05,             # wire box
                                            gymapi.Transform(),
                                            color=(0, 1, 0))
        
        # Get global EE position for visualization
        global_ee_pos_np = self._get_global_ee_pos()
        pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(current_pos[0], current_pos[1], current_pos[2])  # Position above current EE
        pose.p = gymapi.Vec3(global_ee_pos_np[0], global_ee_pos_np[1], global_ee_pos_np[2])  # Global EE Position
        pose.r = gymapi.Quat(current_quat[0], current_quat[1], current_quat[2], current_quat[3])  # Orientation
        # pose.p = gymapi.Vec3(0, 0, 0.5 + 0.25*math.sin(gym.get_sim_time(sim)))
        gymutil.draw_lines(box_geom, gym, viewer, None, pose) 

        # Draw target position sphere
        target_pose = gymapi.Transform()
        target_pose.p = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
        # Visualize with the robot's target orientation
        target_pose.r = gymapi.Quat(target_quat_robot[0], target_quat_robot[1], target_quat_robot[2], target_quat_robot[3])
        gymutil.draw_lines(box_geom_target, gym, viewer, None, target_pose)
        
        # Calculate position delta
        pos_delta = target_pos - global_ee_pos_np
        pos_dist = np.linalg.norm(pos_delta)
        # Print debug information
        self.verbose_print(f"Current pos: {current_pos}, Target pos: {target_pos}, "
                            f"Pos delta: {pos_delta}, Distance: {pos_dist}")
        
        # Calculate orientation delta using the robot's target orientation
        current_rot = st.Rotation.from_quat(current_quat)
        # target_rot is now the desired robot orientation
        target_rot_for_delta = st.Rotation.from_quat(target_quat_robot) 

        # Calculate the full delta rotation
        delta_rotation_object = current_rot.inv() * target_rot_for_delta
        delta_rotvec = delta_rotation_object.as_rotvec()

        # Scale the rotation angle by 0.1
        # Multiplying the rotation vector by a scalar scales its magnitude (the angle of rotation)
        # while keeping the axis of rotation the same.
        scaled_rotvec = delta_rotvec * 0.1 

        # Convert scaled rotvec back to quaternion. This will be a unit quaternion.
        ori_delta_rotation_object = st.Rotation.from_rotvec(scaled_rotvec)
        ori_delta = ori_delta_rotation_object.as_quat()
        # Normalize the quaternion to ensure it's a unit quaternion
        ori_delta /= np.linalg.norm(ori_delta)

        # Print out debug information
        self.verbose_print(f"Current orientation (quat): {current_quat}, "
                            f"Target orientation (quat): {target_quat_robot}, "
                            f"Delta orientation (quat): {ori_delta}")

        # Check if we've reached the waypoint
        # Compare current orientation with target_quat_robot
        # Angle between two quaternions q1 and q2 is 2 * acos(|q1 . q2|)
        # If dot product is close to 1, orientations are similar.
        dot_product = np.abs(np.dot(current_quat, target_quat_robot))
        # Ensure dot_product is not slightly > 1 due to precision
        dot_product = min(dot_product, 1.0)
        angle_diff_rad = 2 * np.arccos(dot_product)

        # Gripper action: close if peg is within grasping distance, open otherwise
        gripper_action = torch.tensor([-1.0 if not self.gripper_closed else 1.0])

            # Create action (position delta + orientation delta + gripper)
        action = torch.cat([
            torch.from_numpy(pos_delta).float(),
            torch.from_numpy(ori_delta).float(), # Use the calculated ori_delta
            gripper_action
        ]).unsqueeze(0).to(self.env.device)

        if pos_dist < self.pos_tolerance and angle_diff_rad < self.ori_tolerance:
            self.verbose_print(f"Waypoint reached. Pos dist: {pos_dist}, Angle diff (rad): {angle_diff_rad}")
            self._advance_state()

            # Remove the visualization lines after reaching the waypoint
            if self.env.viewer is not None:
                gym.clear_lines(self.env.viewer)
        
        return action
    
    def _grasp_action(self, obs) -> torch.Tensor:
        """Close gripper to grasp the peg and hold for a few steps."""
        self.gripper_closed = True
        action = torch.zeros(8).to(self.env.device)
        action[6] = 1.0  # Move down to grasp peg
        action[7] = 1.0  # Close gripper
        
        self.grasp_step_counter += 1
        
        if self.grasp_step_counter >= 7: # Hold grasp for 25 steps
            self.current_state = PlanningState.LIFT_PEG
            self.grasp_step_counter = 0 # Reset counter for next grasp
            self.verbose_print("Grasp complete, moving to LIFT_PEG state.")
        else:
            self.verbose_print(f"Grabbing... step {self.grasp_step_counter}/25")
            # Stay in GRASP_PEG state
            
        return action.unsqueeze(0)
    
    def _lift_action(self, obs) -> torch.Tensor:
        """Lift the peg after grasping."""
        # Move up
        action = torch.zeros(8).to(self.env.device)
        action[2] = self.move_speed  # Move up in Z
        action[6] = 1.0  # Move down to grasp peg
        action[7] = 1.0  # Keep gripper closed

        # Get the current end-effector position in world coordinates
        current_pos = self._get_global_ee_pos()
        self.verbose_print(f"Lifting peg. Current position: {current_pos}, Target height: {self.peg_pose[2] + self.approach_height}")
        
        # Check if lifted enough
        # current_pos_tensor, _ = self.env.get_ee_pose()
        current_height = current_pos[2]  # Z coordinate of the end-effector
        target_lift_height = self.peg_pose[2] + self.approach_height
        remaining_lift_height = target_lift_height - current_height
        
        self.verbose_print(f"Lifting peg. Current height: {current_height:.4f}, Target height: {target_lift_height:.4f}, Remaining: {remaining_lift_height:.4f}")

        if current_height > target_lift_height:
            # Set target for hole approach
            self.target_waypoint = self.hole_pose[:3].copy()
            self.target_waypoint[2] += self.approach_height
            self.target_orientation = self.hole_pose[3:7]
            self.current_state = PlanningState.MOVE_TO_HOLE_APPROACH
            self.verbose_print("Lift complete, moving to MOVE_TO_HOLE_APPROACH state.")
        
        return action.unsqueeze(0)
    
    def _insert_action(self, obs) -> torch.Tensor:
        """Insert peg into hole."""
        # Move down into the hole
        action = torch.zeros(8).to(self.env.device)
        action[2] = -self.move_speed  # Move down
        action[6] = 1.0  # Do not change the orientation
        action[7] = 1.0  # Keep gripper closed
        
        # Check insertion depth
        current_pos = self._get_global_ee_pos()
        if current_pos[2] < self.hole_pose[2] + self.insertion_depth:
            self.current_state = PlanningState.RELEASE_PEG
        
        return action.unsqueeze(0)
    
    def _release_action(self, obs) -> torch.Tensor:
        """Release the peg."""
        self.gripper_closed = False
        self.current_state = PlanningState.RETREAT
        
        # Open gripper
        action = torch.zeros(8).to(self.env.device)
        action[6] = 1.0  # Do not change the orientation
        action[7] = -1.0  # Open gripper
        return action.unsqueeze(0)
    
    def _retreat_action(self, obs) -> torch.Tensor:
        """Move robot away after releasing peg."""
        # Move up
        action = torch.zeros(8).to(self.env.device)
        action[2] = self.move_speed  # Move up
        action[6] = 1.0  # Move down to grasp peg
        action[7] = -1.0  # Keep gripper open
        
        # Check if retreated enough
        current_pos = self._get_global_ee_pos()
        if current_pos[2] > self.hole_pose[2] + 0.01:
            self.current_state = PlanningState.COMPLETE
        
        return action.unsqueeze(0)
    
    def _advance_state(self):
        """Advance to the next planning state."""
        state_transitions = {
            PlanningState.MOVE_TO_PEG_APPROACH: PlanningState.MOVE_TO_PEG,
            PlanningState.MOVE_TO_PEG: PlanningState.GRASP_PEG,
            PlanningState.MOVE_TO_HOLE_APPROACH: PlanningState.INSERT_PEG,
        }
        
        if self.current_state in state_transitions:
            self.current_state = state_transitions[self.current_state]
            self.verbose_print(f"Advanced to state: {self.current_state.name}")
    
    def _store_transition(self, obs, action, reward):
        """Store observation, action, and reward."""
        # obs from FurnitureRLSimEnv (with resize_img=True, np_step_out=False) contains:
        # - color_image1: resized tensor
        # - color_image2: resized_cropped tensor
        # - robot_state: dictionary of tensors
        # - parts_poses: tensor
        # All need to be converted to numpy arrays.
        
        processed_robot_state = {}
        if isinstance(obs["robot_state"], dict):
            for k, v in obs["robot_state"].items():
                if isinstance(v, torch.Tensor):
                    processed_robot_state[k] = v.squeeze().cpu().numpy()
                else:
                    processed_robot_state[k] = v # Should not happen if obs format is consistent
        else:
            # Fallback or error handling if robot_state is not a dict as expected
            processed_robot_state = obs["robot_state"].squeeze().cpu().numpy() if isinstance(obs["robot_state"], torch.Tensor) else obs["robot_state"]

        n_ob = {
            "color_image1": obs["color_image1"].squeeze().cpu().numpy(),
            "color_image2": obs["color_image2"].squeeze().cpu().numpy(),
            "robot_state": processed_robot_state,
            "parts_poses": obs["parts_poses"].squeeze().cpu().numpy(),
        }
        
        self.obs.append(n_ob)
        
        if isinstance(action, torch.Tensor):
            action = action.squeeze().cpu().numpy()
        self.acts.append(action)
        
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        self.rews.append(reward)
        
        # No skills for algorithmic approach
        self.skills.append(0)
    
    def _reset_collector_buffer(self):
        """Reset data collection buffers."""
        self.obs = []
        self.acts = []
        self.rews = []
        self.skills = []
        self.grasp_step_counter = 0 # Reset grasp counter
        self.pause_step_counter = 0 # Reset pause counter
    
    def _save_trajectory(self, success: bool):
        """Save the collected trajectory.""" 
        data = {
            "observations": self.obs,
            "actions": self.acts,
            "rewards": self.rews,
            "skills": self.skills,
            "success": success,
            "furniture": self.furniture,
            "error": False,  # Assuming no specific error tracking for motion planning failures beyond 'success'
            "error_description": "", # Assuming no specific error tracking
        }
        
        # Create save directory
        demo_path = self.data_path / ("success" if success else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename for uncompressed .pkl to match DataCollector format
        base_filename = f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"
        path = demo_path / base_filename
        
        # Save data (pickle_data will save uncompressed if suffix is not .xz)
        pickle_data(data, path)
        self.verbose_print(f"Saved trajectory to {path}")