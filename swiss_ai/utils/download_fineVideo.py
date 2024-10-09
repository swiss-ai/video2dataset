import os
import tarfile
import argparse
import json
import time
from datasets import load_dataset
from tqdm import tqdm

def download_and_tar_videos(num_videos_per_tar, output_dir, download_all=False, max_videos=20):
    """
    Function to download and tar videos in batches.
    """
    # Load dataset in streaming mode
    dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    video_count = 0
    tar_file_count = 0  # This will be the starting 10-digit number for the tar files
    tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
    
    # Open the first tar file
    tar = tarfile.open(tar_filename, 'w')

    # Iterate through the dataset
    for sample in dataset:
        # Break if we reach the limit (for testing purposes)
        if not download_all and video_count == max_videos:
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
            
            
            #print(f"{tar_filename} has been created with {num_videos_per_tar} videos.")
            
            # Update tar filename and open a new tar file (increment the tar file count)
            tar_file_count += 1
            
            if tar_file_count % 5 == 0:
                print(f"{tar_filename} has been created with {num_videos_per_tar} videos.")

            tar_filename = os.path.join(output_dir, f'{tar_file_count:010}.tar')
            tar = tarfile.open(tar_filename, 'w')

    # Ensure the last tar file is closed if it has any remaining videos
    if video_count % num_videos_per_tar != 0:
        tar.close()
        print(f"{tar_filename} has been created with remaining videos.")

def estimate_job_time(num_videos_per_tar, output_dir, num_batches=10):
    """
    Function to estimate time taken for downloading 10 tar batches.
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
        
        # Stop after 10 batches
        if tar_file_count == num_batches:
            break

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download and tar videos from Hugging Face dataset in streaming mode.")
    parser.add_argument('--num_videos_per_tar', default=10, type=int, help="Number of videos to bundle into one tar file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where the tar files will be stored.")
    parser.add_argument('--download_all', action='store_true', help="Whether to download all videos (default is to download only a few for testing).")
    parser.add_argument('--max_videos', type=int, default=1000, help="Max number of videos to download for testing purposes (only applicable if --download_all is not set).")
    parser.add_argument('--estimate', action='store_true', help="Run the job time estimation for 10 tar batches.")
    
    args = parser.parse_args()

    if args.estimate:
        # Run time estimation
        estimate_job_time(args.num_videos_per_tar, args.output_dir)
    else:
        # Run the main function
        print("starting to download")
        download_and_tar_videos(args.num_videos_per_tar, args.output_dir, args.download_all, args.max_videos)
        print("finished succesfully")

## python swiss_ai/utils/download_fineVideo.py --output_dir "/store/swissai/a08/data/raw/finevideo"

