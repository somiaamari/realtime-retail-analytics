import argparse
import os

import cv2


def extract_frames(video_path, output_dir, num_frames=200):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video has 0 frames.")
        return

    print(f"Total frames in video: {total_frames}")
    print(f"Extracting {num_frames} evenly spaced frames to {output_dir}...")

    # Calculate interval to get evenly spaced frames
    interval = max(1, total_frames // num_frames)
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            # Save frame
            frame_name = os.path.join(output_dir, f"calib_frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
            
            if saved_count >= num_frames:
                break
                
        count += 1

    cap.release()
    print(f"Successfully saved {saved_count} images to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames for INT8 calibration"
    )
    parser.add_argument(
        "--video", type=str, default="data/raw/vid.mp4", help="Path to input video"
    )
    parser.add_argument(
        "--out", type=str, default="data/calibration", help="Output directory"
    )
    parser.add_argument(
        "--num", type=int, default=200, help="Number of frames to extract"
    )
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.num)