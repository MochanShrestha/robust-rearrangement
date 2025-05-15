import argparse
import torch
import numpy as np
import random
from pathlib import Path
from omegaconf import OmegaConf

from src.behavior import get_actor
from src.behavior.base import Actor
from src.common.tasks import task2idx
from src.gym import get_rl_env
from src.eval.rollout import calculate_success_rate
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a checkpoint model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--task", type=str, default="one_leg", 
                        choices=["one_leg", "lamp", "round_table", "mug_rack", "factory_peg_hole", "bimanual_insertion"],
                        help="Task to evaluate")
    parser.add_argument("--randomness", type=str, default="low", choices=["low", "med", "high"], 
                        help="Environment randomness level")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--n-rollouts", type=int, default=1, help="Number of evaluation rollouts")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps per rollout")
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment")
    parser.add_argument("--save-rollouts", action="store_true", help="Save rollout data")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--observation-space", type=str, default="state", choices=["state", "image"], 
                        help="Observation space")
    parser.add_argument("--action-type", type=str, default="pos", choices=["pos", "delta", "relative"], 
                        help="Action type")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in checkpoint:
        config = OmegaConf.create(checkpoint["config"])
    else:
        raise ValueError("Checkpoint does not contain configuration information")
    
    # Determine max steps if not provided
    if args.max_steps is None:
        from src.common.tasks import task_timeout
        args.max_steps = task_timeout(args.task)
    
    # Determine rotation representation from config
    # Use the same rotation representation as the model
    # This is crucial to avoid dimension mismatch
    act_rot_repr = config.control.act_rot_repr if hasattr(config.control, "act_rot_repr") else "rot_6d"
    print(f"Using rotation representation: {act_rot_repr}")
    
    # Create environment with matching rotation representation
    print(f"Creating environment for task: {args.task}, randomness: {args.randomness}")
    env = get_rl_env(
        gpu_id=args.gpu,
        task=args.task,
        num_envs=args.n_envs,
        randomness=args.randomness,
        observation_space=args.observation_space,
        action_type=args.action_type,
        act_rot_repr=act_rot_repr,  # Set rotation representation to match model
        headless=not args.visualize,
    )
    
    # Create actor and load weights
    print("Creating actor and loading weights")
    actor: Actor = get_actor(config, device=device)
    
    if "model_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["model_state_dict"])
    else:
        actor.load_state_dict(checkpoint)
    
    actor.eval()
    actor.to(device)
    
    # Set up save directory for rollouts
    rollout_save_dir = None
    if args.save_rollouts:
        from src.common.files import trajectory_save_dir
        rollout_save_dir = trajectory_save_dir(
            controller=config.control.controller if hasattr(config.control, "controller") else "diffik",
            domain="sim",
            task=args.task,
            demo_source="rollout",
            randomness=args.randomness,
            create=True,
        )
    
    # Set the task index
    actor.set_task(task2idx[args.task])
    
    # Run evaluation
    print(f"Running {args.n_rollouts} rollouts...")
    rollout_stats = calculate_success_rate(
        actor=actor,
        env=env,
        n_rollouts=args.n_rollouts,
        rollout_max_steps=args.max_steps,
        epoch_idx=0,
        discount=config.discount if hasattr(config, "discount") else 0.99,
        rollout_save_dir=rollout_save_dir,
        save_rollouts_to_wandb=False,
        save_failures=args.save_rollouts,
        compress_pickles=False,
        resize_video=True,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Success Rate: {rollout_stats.success_rate:.2%} ({rollout_stats.n_success}/{rollout_stats.n_rollouts})")
    print(f"Average Return: {rollout_stats.total_return / rollout_stats.n_rollouts:.4f}")
    print(f"Total Reward: {rollout_stats.total_reward:.4f}")
    
    if rollout_save_dir:
        print(f"Rollouts saved to: {rollout_save_dir}")

if __name__ == "__main__":
    main()