import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr


def get_episode_info(zarr_path: str) -> Dict:
    """
    Get episode information from a zarr file.
    
    Args:
        zarr_path: Path to the zarr directory
    
    Returns:
        Dictionary containing:
            - total_episodes: Total number of episodes
            - total_timesteps: Total number of timesteps
            - episodes: List of dictionaries with episode details
                - episode_idx: Episode index
                - steps: Number of steps in this episode
                - offset: Starting offset of this episode in the timestep arrays
                - task: Task name for this episode
                - success: Success flag (1 for success, 0 for failure)
                - pickle_file: Original pickle file path
    """
    # Open the zarr store
    store = zarr.open(zarr_path, mode='r')
    
    # Get episode_ends array which contains cumulative indices where episodes end
    episode_ends = store['episode_ends'][:]
    
    # Get total number of episodes
    total_episodes = len(episode_ends)
    
    # Get tasks and success flags if available
    tasks = store['task'][:] if 'task' in store else ['unknown'] * total_episodes
    successes = store['success'][:] if 'success' in store else np.zeros(total_episodes)
    pickle_files = store['pickle_file'][:] if 'pickle_file' in store else ['unknown'] * total_episodes
    
    # Calculate number of steps and offset for each episode
    episodes = []
    prev_end = 0
    
    for i, end in enumerate(episode_ends):
        steps = end - prev_end
        offset = prev_end
        
        episode_info = {
            'episode_idx': i,
            'steps': int(steps),
            'offset': int(offset),
            'task': tasks[i],
            'success': int(successes[i]),
            'pickle_file': pickle_files[i]
        }
        
        episodes.append(episode_info)
        prev_end = end
    
    # Create the result dictionary
    result = {
        'total_episodes': total_episodes,
        'total_timesteps': int(episode_ends[-1]) if total_episodes > 0 else 0,
        'episodes': episodes,
        'metadata': dict(store.attrs)
    }
    
    return result


def print_episode_summary(zarr_path: str, detailed: bool = False, episode_idx: int = None):
    """
    Print a summary of episode information from a zarr file.
    
    Args:
        zarr_path: Path to the zarr directory
        detailed: Whether to print detailed information for each episode
        episode_idx: If specified, only print details for this episode index
    """
    info = get_episode_info(zarr_path)
    
    print(f"Zarr file: {zarr_path}")
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total timesteps: {info['total_timesteps']}")
    
    if 'metadata' in info:
        print("\nMetadata:")
        for key, value in info['metadata'].items():
            print(f"  {key}: {value}")
    
    if detailed:
        print("\nEpisode details:")
        for episode in info['episodes']:
            if episode_idx is not None and episode['episode_idx'] != episode_idx:
                continue
                
            print(f"\nEpisode {episode['episode_idx']}:")
            print(f"  Steps: {episode['steps']}")
            print(f"  Offset: {episode['offset']}")
            print(f"  Task: {episode['task']}")
            print(f"  Success: {episode['success']}")
            print(f"  Pickle file: {episode['pickle_file']}")
    
    # If episode_idx was specified but no matching episode was found
    if episode_idx is not None and all(episode['episode_idx'] != episode_idx for episode in info['episodes']):
        print(f"\nNo episode with index {episode_idx} found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get episode information from a zarr file")
    parser.add_argument("zarr_path", type=str, help="Path to the zarr directory")
    parser.add_argument("--detailed", "-d", action="store_true", help="Print detailed information for each episode")
    parser.add_argument("--episode", "-e", type=int, default=None, help="Print details for a specific episode index")
    
    args = parser.parse_args()
    print_episode_summary(args.zarr_path, args.detailed, args.episode)
