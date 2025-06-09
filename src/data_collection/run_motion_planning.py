#!/usr/bin/env python3
"""
Script to run motion planning data collection for peg-in-hole task.
"""

import argparse
import numpy as np
from pathlib import Path

from motion_planning_collector import MotionPlanningCollector

# Add your project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.common.files import trajectory_save_dir
    from furniture_bench.envs.initialization_mode import Randomness
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    print("And that the conda environment is activated")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Motion Planning Data Collection")
    parser.add_argument("--furniture", "-f", type=str, default="factory_peg_hole", 
                       help="Furniture type (should be factory_peg_hole for this planner)")
    parser.add_argument("--randomness", "-r", type=str, default="low", 
                       choices=["low", "med", "high"],
                       help="Environment randomness level")
    parser.add_argument("--num-demos", "-n", type=int, default=50,
                       help="Number of demonstrations to collect")
    parser.add_argument("--gpu-id", "-g", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode (no GUI)")
    parser.add_argument("--save-failure", action="store_true",
                       help="Save failed trajectories")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Custom data directory (optional)")
    
    # Motion planning specific arguments
    parser.add_argument("--max-velocity", type=float, default=0.1,
                       help="Maximum robot velocity")
    parser.add_argument("--approach-height", type=float, default=0.05,
                       help="Height above objects for approach")
    parser.add_argument("--position-tolerance", type=float, default=0.0001,
                       help="Position tolerance for waypoints")
    
    args = parser.parse_args()
    
    # Validate furniture type
    if args.furniture != "factory_peg_hole":
        print(f"Warning: This planner is designed for factory_peg_hole, got {args.furniture}")
    
    # Set up data directory using the correct parameters
    if args.data_dir:
        data_path = Path(args.data_dir)
    else:
        try:
            data_path = trajectory_save_dir(
                controller="diffik",
                domain="sim", 
                task=args.furniture,
                demo_source="motion_planning",
                randomness=args.randomness,
            )
        except Exception as e:
            print(f"Error creating data directory: {e}")
            print("Using fallback directory...")
            data_path = Path(f"./data/motion_planning/{args.furniture}/{args.randomness}")
            data_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Data will be saved to: {data_path}")
    
    # Create motion planning collector
    try:
        collector = MotionPlanningCollector(
            data_path=str(data_path),
            furniture=args.furniture,
            randomness=Randomness.str_to_enum(args.randomness),
            compute_device_id=args.gpu_id,
            graphics_device_id=args.gpu_id,
            headless=args.headless,
            num_demos=args.num_demos,
            save_failure=args.save_failure,
            verbose=args.verbose,
        )
        
        # Update planner parameters
        collector.move_speed = args.max_velocity / 10  # Scale for per-step movement
        collector.approach_height = args.approach_height
        collector.pos_tolerance = args.position_tolerance
        
    except Exception as e:
        print(f"Error creating MotionPlanningCollector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Starting motion planning data collection...")
    print(f"Target: {args.num_demos} successful demonstrations")
    print(f"Randomness: {args.randomness}")
    print(f"Max velocity: {args.max_velocity} m/s")
    print(f"Approach height: {args.approach_height} m")
    
    try:
        collector.collect()
        print("\nData collection completed successfully!")
        print(f"Collected {collector.num_success} successful trajectories")
        print(f"Failed trajectories: {collector.num_fail}")
        print(f"Success rate: {collector.num_success / collector.traj_counter * 100:.1f}%")
        
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
        print(f"Collected {collector.num_success} successful trajectories so far")
        
    except Exception as e:
        print(f"\nError during data collection: {e}")
        print(f"Collected {collector.num_success} successful trajectories before error")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()