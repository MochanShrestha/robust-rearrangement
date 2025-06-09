#!/usr/bin/env python3
"""Replay recorded demonstrations from pickle files."""

import argparse
import pickle
import time
from pathlib import Path

import furniture_bench
import gym
import torch
import numpy as np
from tqdm import tqdm

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
from furniture_bench.envs.observation import FULL_OBS
from furniture_bench.envs.initialization_mode import Randomness


def main():
    parser = argparse.ArgumentParser(description="Replay recorded demonstrations")
    parser.add_argument("--furniture", default="one_leg", help="Default furniture if not found in demo file")
    parser.add_argument(
        "--replay-path", 
        type=str, 
        required=True,
        help="Path to the saved demonstration pickle file to replay"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        help="Directory containing multiple demo files to replay sequentially"
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl*",
        help="File pattern to match when using --demo-dir"
    )
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument(
        "--speed-multiplier",
        type=float,
        default=-1.0,
        help="Playback speed multiplier (1.0 = normal speed)"
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment"
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness"
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSimFull-v0",
        help="Environment id"
    )
    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space",
        choices=["quat", "axis", "rot_6d"],
        default="quat"
    )
    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation"
    )
    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering"
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the video of the simulator"
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator"
    )
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        default="diffik",
        choices=["osc", "diffik"],
        help="Control mode for the robot"
    )
    parser.add_argument(
        "--ee-laser",
        action="store_true",
        help="Show laser from end-effector"
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=3000,
        help="Maximum environment steps"
    )
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Pause after each action (press Enter to continue)"
    )
    parser.add_argument(
        "--show-info",
        action="store_true",
        help="Show detailed information about each step"
    )

    args = parser.parse_args()

    if args.demo_dir:
        replay_directory(args)
    else:
        replay_single_demo(args)


def replay_single_demo(args):
    """Replay a single demonstration file."""
    if not args.replay_path:
        raise ValueError("Must specify --replay-path or --demo-dir")
    
    replay_path = Path(args.replay_path)
    if not replay_path.exists():
        raise FileNotFoundError(f"Demo file not found: {replay_path}")

    # Load demonstration data
    furniture_name = args.furniture
    try:
        with open(replay_path, "rb") as f:
            data = pickle.load(f)
            
        if "furniture" in data:
            furniture_name = data["furniture"]
            print(f"Using furniture '{furniture_name}' from demo file: {replay_path}")
        else:
            print(f"Warning: 'furniture' field not found in demo file. Using default: {furniture_name}")
            
    except Exception as e:
        print(f"Error loading demo file {replay_path}: {e}. Using default furniture: {furniture_name}")
        return

    # Extract demo information
    observations = data.get("observations", [])
    actions = data.get("actions", [])
    rewards = data.get("rewards", [])
    success = data.get("success", False)
    
    print(f"Demo info:")
    print(f"  File: {replay_path}")
    print(f"  Furniture: {furniture_name}")
    print(f"  Success: {success}")
    print(f"  Trajectory length: {len(observations)} observations, {len(actions)} actions")
    total_reward = sum(r for r in rewards if r is not None) if rewards else 0
    print(f"  Total reward: {total_reward if rewards else 'N/A'}")

    # Create environment using FurnitureRLSimEnv directly (matches data collection setup)
    randomness_enum = Randomness.str_to_enum(args.randomness)
    
    env = FurnitureRLSimEnv(
        furniture=furniture_name,
        obs_keys=FULL_OBS,
        headless=args.headless,
        max_env_steps=args.max_env_steps,
        num_envs=args.num_envs,
        act_rot_repr=args.act_rot_repr,
        action_type="delta",
        manual_done=True,
        resize_img=True,#not args.high_res,
        np_step_out=False,
        channel_first=False,
        randomness=randomness_enum,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
    )

    def action_tensor(ac):
        """Convert action to tensor format expected by environment."""
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)
        
        ac = ac.clone() if isinstance(ac, torch.Tensor) else torch.tensor(ac)
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Initialize environment
    env.reset()
    
    # Reset to initial state if available
    if len(observations) > 0:
        try:
            env.refresh()
            env.reset_env_to(env_idx=0, state=observations[0]) # Changed from observations[10]
            print("Reset environment to initial demonstration state")
        except AttributeError:
            print("Warning: Environment doesn't support reset_env_to, starting from default reset")
        except Exception as e:
            print(f"Warning: Could not reset to initial state: {e}")

    # Replay the demonstration
    print(f"\nStarting replay...")
    print("Press Ctrl+C to stop replay\n")
    
    try:
        pbar = tqdm(total=len(actions), desc="Replaying")
        
        for i, action in enumerate(actions):
            if args.show_info:
                print(f"Step {i+1}/{len(actions)}: action = {action}")
                if i < len(rewards):
                    reward_val = rewards[i] if rewards[i] is not None else "None"
                    print(f"  Reward: {reward_val}")
            
            # Convert action to tensor
            action_t = action_tensor(action)
            
            # Step environment
            ob, rew, done, info = env.step(action_t)
            
            # Add delay for visualization
            if not args.headless and args.speed_multiplier > 0:
                time.sleep(0.1 / args.speed_multiplier)
            
            # Step-by-step mode
            if args.step_by_step:
                input(f"Step {i+1} completed. Press Enter to continue...")
            
            pbar.update(1)
            
            if done:
                print(f"\nEpisode finished early at step {i+1}")
                break
        
        pbar.close()
        
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    
    print("Replay completed!")
    env.close()


def replay_directory(args):
    """Replay multiple demonstrations from a directory."""
    demo_dir = Path(args.demo_dir)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")
    
    # Find all demo files
    demo_files = list(demo_dir.glob(args.pattern))
    if not demo_files:
        print(f"No demo files found matching pattern '{args.pattern}' in {demo_dir}")
        return
    
    demo_files.sort()  # Sort for consistent ordering
    print(f"Found {len(demo_files)} demonstration files")
    
    try:
        for i, demo_file in enumerate(demo_files):
            print(f"\n{'='*60}")
            print(f"Replaying demo {i+1}/{len(demo_files)}: {demo_file.name}")
            print(f"{'='*60}")
            
            # Temporarily override replay_path for single demo function
            args.replay_path = str(demo_file)
            replay_single_demo(args)
            
            # Ask user if they want to continue (except for last file)
            if i < len(demo_files) - 1 and not args.headless:
                response = input("\nContinue to next demo? (y/n/q): ").lower()
                if response == 'n':
                    break
                elif response == 'q':
                    print("Quitting...")
                    break
                    
    except KeyboardInterrupt:
        print("\n\nDirectory replay interrupted by user")
    
    print(f"\nCompleted replaying {min(i+1, len(demo_files))} demonstrations")


if __name__ == "__main__":
    main()