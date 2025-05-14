import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import zarr
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.visualization.render_mp4 import unpickle_data
from src.data_processing.get_num_episodes import get_episode_info


def get_robot_state_feature_names():
    """
    Return a list of feature names for the robot state.
    
    The robot state uses 6D rotation representation (converted from quaternions).
    """
    # Based on the robot state data structure in process_pickles.py
    # First 3 values are position, next 6 are rotation (6D representation), 
    # followed by gripper state and other proprioceptive features
    feature_names = [
        # Position
        "X (m)", 
        "Y (m)", 
        "Z (m)",
        
        # Rotation in 6D representation (converted from quaternion)
        "R1", 
        "R2", 
        "R3",
        "R4", 
        "R5", 
        "R6",
        
        # Gripper state
        "Gripper (m)",
        
        # Joint states and other proprioceptive data
        "J1 (rad)",
        "J2 (rad)",
        "J3 (rad)",
        "J4 (rad)",
        "J5 (rad)",
        "J6 (rad)",
        "J7 (rad)",
    ]
    
    # Return a function that maps index to name, handling out of range indices
    def get_name(idx):
        if idx < len(feature_names):
            return feature_names[idx]
        else:
            return f"Feature {idx}"
    
    return get_name


def generate_robot_state_graphs(robot_state_data, output_path, title=None):
    """
    Generate graphs of robot state data over time.
    
    Args:
        robot_state_data: Array of robot state data with shape [timesteps, features]
        output_path: Path to save the output PNG file
        title: Optional title for the plot
    """
    n_timesteps, n_features = robot_state_data.shape
    time_steps = np.arange(n_timesteps)
    
    # Get feature name mapping function
    get_feature_name = get_robot_state_feature_names()
    
    # Create a figure with one subplot per feature
    plt.figure(figsize=(14, n_features * 1.5))
    plt.subplots_adjust(hspace=0.5)
    
    # Group features for cleaner visualization
    feature_groups = {
        'Position': [0, 1, 2],
        'Rotation': [3, 4, 5, 6, 7, 8],
        'Gripper': [9],
        'Joints': list(range(10, min(17, n_features))),
        'Other': list(range(17, n_features))
    }
    
    # Plot each feature in its own subplot
    for i in range(n_features):
        ax = plt.subplot(n_features, 1, i + 1)
        
        # Determine the group this feature belongs to for color coding
        group_name = next((name for name, indices in feature_groups.items() if i in indices), 'Other')
        
        # Color coding based on feature group
        color_map = {
            'Position': 'tab:blue',
            'Rotation': 'tab:orange',
            'Gripper': 'tab:green',
            'Joints': 'tab:red',
            'Other': 'tab:purple'
        }
        
        plt.plot(time_steps, robot_state_data[:, i], color=color_map.get(group_name, 'tab:blue'))
        plt.ylabel(get_feature_name(i), fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a background color to visually group related features
        # Use proper alpha value instead of appending "15" to the color name
        if group_name != 'Other':
            ax.set_facecolor(color_map[group_name])
            ax.patch.set_alpha(0.1)  # Set transparency properly
            
        # Only add x-axis label to the bottom subplot
        if i == n_features - 1:
            plt.xlabel('Time Steps')
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Robot state graphs saved to {output_path}")


def visualize_episode_from_zarr(zarr_path: str, episode_idx: int = 0, output_dir: str = None):
    """
    Generate a video visualization from a zarr file for a specific episode.
    
    Args:
        zarr_path: Path to the zarr directory
        episode_idx: Index of the episode to visualize
        output_dir: Optional output directory. If None, the video will be saved
                    in the same directory as the zarr file.
    """
    # Open the zarr store
    print(f"Loading zarr data from {zarr_path}")
    store = zarr.open(zarr_path, mode='r')
    
    # Get episode information
    episode_info = get_episode_info(zarr_path)
    
    # Check if episode exists
    if episode_idx >= episode_info['total_episodes']:
        raise ValueError(f"Episode {episode_idx} not found. Total episodes: {episode_info['total_episodes']}")
    
    # Get episode details
    episode = next(ep for ep in episode_info['episodes'] if ep['episode_idx'] == episode_idx)
    start_idx = episode['offset']
    end_idx = start_idx + episode['steps']
    
    # Create output path
    zarr_dir = Path(zarr_path)
    zarr_name = zarr_dir.name.replace('.zarr', '')
    
    if output_dir is None:
        # Default to same directory as zarr file
        output_dir = zarr_dir.parent
    else:
        output_dir = Path(output_dir)
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filenames
    output_filename = f"{zarr_name}_episode_{episode_idx}.mp4"
    output_path = output_dir / output_filename
    
    # Create output path for robot state graphs
    graph_filename = f"{zarr_name}_episode_{episode_idx}_robot_state.png"
    graph_path = output_dir / graph_filename
    
    print(f"Generating video for episode {episode_idx} (steps {start_idx}-{end_idx}) at {output_path}")
    
    # Extract robot state data for this episode
    robot_state = store['robot_state'][start_idx:end_idx]
    
    # Generate robot state graphs
    title = f"Robot State - {zarr_name} - Episode {episode_idx}"
    generate_robot_state_graphs(robot_state, graph_path, title)
    
    # Get image dimensions from first frame
    img1 = store['color_image1'][start_idx]
    img2 = store['color_image2'][start_idx]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make sure heights are the same for side-by-side concatenation
    max_h = max(h1, h2)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w1 + w2, max_h))
    
    # Process each frame for this episode
    for i in tqdm(range(start_idx, end_idx), desc="Creating video"):
        # Get both camera images
        img1 = store['color_image1'][i]
        img2 = store['color_image2'][i]
        
        # Resize images to match height if needed
        if h1 != max_h:
            img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
        if h2 != max_h:
            img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Join images side by side
        combined_img = np.hstack((img1_rgb, img2_rgb))
        
        # Write to video file
        out.write(combined_img)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")
    
    # Display success information
    duration = episode['steps'] / 10.0  # assuming 10 fps
    print(f"Video duration: {duration:.2f} seconds ({episode['steps']} frames)")
    return output_path, graph_path


def visualize_episode_from_pickle(pickle_path: str, output_dir: str = None):
    """
    Generate a video visualization from a pickle file.
    
    Args:
        pickle_path: Path to the pickle file
        output_dir: Optional output directory. If None, the video will be saved
                    in the same directory as the pickle file.
    """
    # Load pickle data
    print(f"Loading data from {pickle_path}")
    data = unpickle_data(Path(pickle_path))
    
    # Extract observations
    observations = data["observations"]
    
    # Check if we have the expected camera image keys
    if "color_image1" not in observations[0] or "color_image2" not in observations[0]:
        raise ValueError("Pickle file doesn't contain expected camera images")
    
    # Create output paths
    pickle_path = Path(pickle_path)
    pickle_name = pickle_path.stem
    
    if output_dir is None:
        # Default to same directory as pickle file
        output_path = pickle_path.with_suffix('.mp4')
        graph_path = pickle_path.with_suffix('.robot_state.png')
    else:
        # Use specified directory with pickle filename
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / pickle_path.with_suffix('.mp4').name
        graph_path = output_dir / f"{pickle_name}.robot_state.png"
    
    print(f"Generating video at {output_path}")
    
    # Extract robot state data
    if "robot_state" in observations[0]:
        if isinstance(observations[0]["robot_state"], dict):
            # For structured robot states, we need to flatten them
            from furniture_bench.robot.robot_state import filter_and_concat_robot_state
            robot_state = np.array([filter_and_concat_robot_state(o["robot_state"]) 
                                    for o in observations], dtype=np.float32)
        else:
            # For already flattened robot states
            robot_state = np.array([o["robot_state"] for o in observations], dtype=np.float32)
            
        # Generate robot state graphs
        title = f"Robot State - {pickle_name}"
        generate_robot_state_graphs(robot_state, graph_path, title)
    else:
        print("Warning: No robot state data found in pickle file")
    
    # Get image dimensions from first frame
    img1 = observations[0]["color_image1"]
    img2 = observations[0]["color_image2"]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make sure heights are the same for side-by-side concatenation
    max_h = max(h1, h2)
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w1 + w2, max_h))
    
    # Process each frame
    for i in tqdm(range(len(observations)), desc="Creating video"):
        # Get both camera images
        img1 = observations[i]["color_image1"]
        img2 = observations[i]["color_image2"]
        
        # Resize images to match height if needed
        if h1 != max_h:
            img1 = cv2.resize(img1, (int(w1 * max_h / h1), max_h))
        if h2 != max_h:
            img2 = cv2.resize(img2, (int(w2 * max_h / h2), max_h))
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Join images side by side
        combined_img = np.hstack((img1_rgb, img2_rgb))
        
        # Write to video file
        out.write(combined_img)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")
    
    # Display success information
    duration = len(observations) / 10.0  # assuming 10 fps
    print(f"Video duration: {duration:.2f} seconds ({len(observations)} frames)")
    return output_path, graph_path


def visualize_episode(file_path: str, episode_idx: int = None, output_dir: str = None):
    """
    Generate a video visualization and robot state graphs from a file (zarr or pickle).
    
    Args:
        file_path: Path to the zarr directory or pickle file
        episode_idx: Index of the episode to visualize (only for zarr files)
        output_dir: Optional output directory for the video. If None, the video will
                    be saved in the same directory as the input file.
                    
    Returns:
        Tuple of (video_path, graph_path) or just video_path if no robot state data
    """
    file_path = str(file_path)  # Convert Path to string if necessary
    
    # Check if the file is a zarr directory or pickle file
    if os.path.isdir(file_path) or file_path.endswith('.zarr'):
        # It's a zarr directory
        if episode_idx is None:
            episode_idx = 0
            print(f"No episode index specified. Defaulting to episode {episode_idx}.")
        return visualize_episode_from_zarr(file_path, episode_idx, output_dir)
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        # It's a pickle file
        if episode_idx is not None:
            print("Warning: Episode index is ignored for pickle files as they contain a single episode.")
        return visualize_episode_from_pickle(file_path, output_dir)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Must be a zarr directory or pickle file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video and robot state graphs from a zarr or pickle file")
    parser.add_argument(
        "file_path", 
        type=str,
        help="Path to the zarr directory or pickle file"
    )
    parser.add_argument(
        "--episode", 
        "-e", 
        type=int,
        default=None,
        help="Episode index to visualize (only applicable for zarr files)"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        type=str,
        default=None,
        help="Directory to save the output files (default: same directory as input)"
    )
    
    args = parser.parse_args()
    result = visualize_episode(args.file_path, args.episode, args.output_dir)
    
    # Print summary of output files
    if isinstance(result, tuple):
        video_path, graph_path = result
        print(f"\nGenerated files:")
        print(f"  Video: {video_path}")
        print(f"  Robot state graphs: {graph_path}")
    else:
        print(f"\nGenerated video: {result}")
