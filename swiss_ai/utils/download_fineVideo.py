import os
import tarfile
import argparse
import json
import time
from datasets import load_dataset
from tqdm import tqdm
import sys


def skip_samples(dataset, num_to_skip):
    """
    Skip the first num_to_skip samples from the dataset iterator.
    """
    for _ in tqdm(range(num_to_skip), desc="Skipping samples"):
        next(dataset)


def download_and_tar_videos(num_videos_per_tar, output_dir, download_all=False, max_videos=20, start_from_tar=0):
    """
    Function to download and tar videos in batches, with an option to resume from a specified tar file.
    """
    # Load dataset in streaming mode
    dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print('starting to download to', output_dir)
    
    sys.stdout.flush()
    # Initialize variables
    video_count = 0 # start_from_tar * num_videos_per_tar  # Calculate the video count from the tar number
    tar_file_count = start_from_tar  # Start from the specified tar file number
    tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
    
    # Calculate how many videos to skip based on tar file and videos per tar
    num_videos_to_skip = start_from_tar * num_videos_per_tar
    start_time = time.time()


    # Open a new tar file if continuing from the specified tar
    tar = tarfile.open(tar_filename, 'w')
    
    # Iterate through the dataset
    for sample in dataset:
        # Skip samples until reaching the starting video count
        if video_count < num_videos_to_skip:
            # 
            if video_count % 100 == 0:
                curr_time = time.time()
                print(f"Skipped {video_count} samples in {curr_time - start_time:.2f} seconds")
                sys.stdout.flush()
            
            video_count += 1
            continue
        
        # Break if we reach the limit (for testing purposes)
        if not download_all and video_count >= max_videos:
            break
        
        # Define 16-digit filenames for video and JSON metadata
        video_filename = f'{video_count:016}.mp4'
        json_filename = f'{video_count:016}.json'
        
        # Write JSON data to file
        with open(json_filename, 'w') as text_file:
            text_file.write(json.dumps(sample['json']))
        
        # Write video data to file
        with open(video_filename, 'wb') as video_file:
            video_file.write(sample['mp4'])
        
        # Add the files to the tar archive
        tar.add(json_filename)
        tar.add(video_filename)
        
        # Remove the files from the file system (to avoid storing too much data locally)
        os.remove(json_filename)
        os.remove(video_filename)
        
        #print(f"Saved and added sample {video_count} to {tar_filename}")
        
        video_count += 1
        
        # Close current tar and start a new one when reaching num_videos_per_tar
        if video_count % num_videos_per_tar == 0:
            tar.close()
            print(f"{tar_filename} has been created with {num_videos_per_tar} videos.")
            sys.stdout.flush()
            # Update tar filename and open a new tar file (increment the tar file count)
            tar_file_count += 1
            tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
            tar = tarfile.open(tar_filename, 'w')

    # Ensure the last tar file is closed if it has any remaining videos
    if video_count % num_videos_per_tar != 0:
        tar.close()
        print(f"{tar_filename} has been created with remaining videos.")
        sys.stdout.flush()
        
def estimate_job_time(num_videos_per_tar, output_dir, num_batches=20):
    """
    Function to estimate time taken for downloading 20 tar batches.
    """
    # Load dataset in streaming mode
    dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    video_count = 0
    tar_file_count = 0  # Starting 10-digit tar file number
    tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
    
    # Open the first tar file
    tar = tarfile.open(tar_filename, 'w')
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(dataset, desc="Estimating time")):
        # Save video and JSON metadata
        video_filename = f'{video_count:016}.mp4'
        json_filename = f'{video_count:016}.json'
        
        with open(json_filename, 'w') as text_file:
            text_file.write(json.dumps(sample['json']))
        with open(video_filename, 'wb') as video_file:
            video_file.write(sample['mp4'])
        
        tar.add(json_filename)
        tar.add(video_filename)
        
        os.remove(json_filename)
        os.remove(video_filename)
        
        video_count += 1
        
        if video_count % num_videos_per_tar == 0:
            tar.close()
            batch_time = time.time() - start_time
            print(f"Batch {tar_file_count:010} completed in {batch_time:.2f} seconds")
            tar_file_count += 1
            tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
            tar = tarfile.open(tar_filename, 'w')
            
            start_time = time.time()
        
        # Stop after 20 batches
        if tar_file_count == num_batches:
            break

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and tar videos from Hugging Face dataset in streaming mode.")
    parser.add_argument('--num_videos_per_tar', default=4, type=int, help="Number of videos to bundle into one tar file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the tar files will be stored.")
    parser.add_argument('--download_all', action='store_true', help="Whether to download all videos (default is to download only a few for testing).")
    parser.add_argument('--max_videos', type=int, default=10, help="Max number of videos to download for testing purposes (only applicable if --download_all is not set).")
    parser.add_argument('--estimate', action='store_true', help="Run the job time estimation for 20 tar batches.")
    parser.add_argument('--start_from_tar', type=int, default=0, help="Start downloading from a specific tar file number (default is 0).")
    
    args = parser.parse_args()

    if args.estimate:
        # Run time estimation
        estimate_job_time(args.num_videos_per_tar, args.output_dir)
    else:
        # Run the main function with the option to start from a specific tar
        download_and_tar_videos(args.num_videos_per_tar, args.output_dir, args.download_all, args.max_videos, args.start_from_tar)