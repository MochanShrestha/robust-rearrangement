import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# ANSI escape codes for colors
RED = '\033[91m'
RESET = '\033[0m'

def download_s3_directory(s3_client, bucket, s3_path, local_path):
    """
    Download a directory from S3 to a local path
    Returns True if files were found and downloaded, False otherwise.
    """
    found_files = False
    paginator = s3_client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=s3_path):
        if "Contents" in result:
            for key in result["Contents"]:
                found_files = True  # Mark that at least one file/object is found
                relative_path = os.path.relpath(key["Key"], s3_path)
                local_file_path = os.path.join(local_path, relative_path)
                # Ensure the local directory for the file exists
                if relative_path == ".": # Skip if it's the directory placeholder itself
                    continue
                
                # Create directory if it's a file and not just a prefix
                if key["Key"].endswith("/"): # Skip if it's a "directory" object in S3
                    os.makedirs(local_file_path, exist_ok=True)
                    print(f"Created directory: {local_file_path}")
                else:
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    s3_client.download_file(bucket, key["Key"], local_file_path)
                    print(f"Downloaded: {key['Key']} to {local_file_path}")
    
    if not found_files:
        print(f"{RED}Error: No files found at s3://{bucket}/{s3_path}{RESET}")
    
    return found_files

def main():
    parser = argparse.ArgumentParser(description="Download data from S3 for a specific task.")
    parser.add_argument("--task", required=True, help="Task name (e.g., lamp, one_leg, round_table)")
    parser.add_argument("--type", default="processed", choices=["processed", "raw", "checkpoints"], help="Type of data to download (processed, raw, or checkpoints). Default is 'processed'.")
    parser.add_argument("--checkpoint_type", choices=["bc", "rppo"], help="Type of checkpoint model (bc or rppo). Required if --type is checkpoints.")
    parser.add_argument("--checkpoint_level", choices=["low", "med"], help="Level of checkpoint (low or med). Required if --type is checkpoints.")
    args = parser.parse_args()

    # S3 bucket
    bucket = "iai-robust-rearrangement"
    data_dir = os.getenv("DATA_DIR_PROCESSED", "./data") # Note: DATA_DIR_PROCESSED might need a more generic name if handling raw data too.

    if args.type == "processed":
        base_path = f"data/processed/diffik/sim/{args.task}/teleop"
        local_dir = os.path.join(data_dir, "processed", "diffik", "sim", args.task, "teleop")
    elif args.type == "raw":
        base_path = f"data/raw/diffik/sim/{args.task}/rollout/"
        local_dir = os.path.join(data_dir, "raw", "diffik", "sim", args.task, "rollout")
    elif args.type == "checkpoints":
        if not args.checkpoint_type or not args.checkpoint_level:
            print(f"{RED}Error: --checkpoint_type and --checkpoint_level are required when --type is 'checkpoints'.{RESET}")
            return
        base_path = f"checkpoints/{args.checkpoint_type}/{args.task}/{args.checkpoint_level}"
        local_dir = os.path.join(data_dir, "checkpoints", args.checkpoint_type, args.task, args.checkpoint_level)
    else:
        # This case should not be reached due to 'choices' in argparse, but good for safety.
        print(f"{RED}Error: Unknown data type '{args.type}'.{RESET}")
        return

    # Create S3 client without requiring credentials (for public buckets)
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    print(f"Downloading files for task: {args.task}")
    print(f"From S3 path: s3://{bucket}/{base_path}")
    print(f"To local directory: {local_dir}")

    files_downloaded = download_s3_directory(s3_client, bucket, base_path, local_dir)

    if files_downloaded:
        print("Download complete.")
    else:
        # Error message is printed by download_s3_directory
        print("Download process finished.")


if __name__ == "__main__":
    main()