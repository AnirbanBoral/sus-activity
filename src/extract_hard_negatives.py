import cv2
import os

def extract_frames(video_path, output_dir, prefix, sample_every=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % sample_every == 0:
            frame_filename = os.path.join(output_dir, f"{prefix}_frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"Extracted frames for {prefix}: {count} total processed, saved {saved} frames to {output_dir}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manual_test_dir = os.path.join(script_dir, 'manual_test')
    output_dir = os.path.join(script_dir, 'data', 'normal', 'Train_HardNegatives')
    
    # Extract from all available normal-looking videos, sampling every frame for denser coverage
    videos = [
        ('running.avi', 'run', 2),           # every 2nd frame
        ('walking.avi', 'walk', 1),           # every frame — most important to fix
        ('Human-Activity.avi', 'human', 2),   # general normal human activity
        ('vid.mp4', 'vid', 1),                # all frames from this short clip
    ]
    
    for video_file, prefix, sample_every in videos:
        video_path = os.path.join(manual_test_dir, video_file)
        if os.path.exists(video_path):
            print(f"Processing {video_path}...")
            extract_frames(video_path, output_dir, prefix, sample_every)
        else:
            print(f"Warning: {video_path} does not exist.")
