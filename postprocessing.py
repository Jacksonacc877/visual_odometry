import cv2
import numpy as np
import os

def combine_videos(video1_path, video2_path, output_path, side_by_side=True):
    """
    Combines two videos into one.
    
    Args:
        video1_path (str): Path to the first video
        video2_path (str): Path to the second video
        output_path (str): Path to save the combined video
        side_by_side (bool): If True, videos are combined horizontally; otherwise, vertically
    """
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Check if videos opened successfully
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos.")
        return
    
    # Get video properties
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    # Calculate output video dimensions
    if side_by_side:
        out_width = width1 + width2
        out_height = max(height1, height2)
    else:
        out_width = max(width1, width2)
        out_height = height1 + height2
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    while True:
        # Read frames from both videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Break if either video ends
        if not ret1 or not ret2:
            break
        
        # Resize frames if they have different dimensions
        frame1 = cv2.resize(frame1, (width1, height1))
        frame2 = cv2.resize(frame2, (width2, height2))
        
        # Combine frames
        if side_by_side:
            # If heights are different, pad the shorter frame
            if height1 != height2:
                if height1 < height2:
                    pad = np.zeros((height2 - height1, width1, 3), dtype=np.uint8)
                    frame1 = np.vstack((frame1, pad))
                else:
                    pad = np.zeros((height1 - height2, width2, 3), dtype=np.uint8)
                    frame2 = np.vstack((frame2, pad))
            
            combined_frame = np.hstack((frame1, frame2))
        else:
            # If widths are different, pad the narrower frame
            if width1 != width2:
                if width1 < width2:
                    pad = np.zeros((height1, width2 - width1, 3), dtype=np.uint8)
                    frame1 = np.hstack((frame1, pad))
                else:
                    pad = np.zeros((height2, width1 - width2, 3), dtype=np.uint8)
                    frame2 = np.hstack((frame2, pad))
            
            combined_frame = np.vstack((frame1, frame2))
        
        # Write the combined frame
        out.write(combined_frame)
    
    # Release resources
    cap1.release()
    cap2.release()
    out.release()
    print(f"Combined video saved to {output_path}")

if __name__ == "__main__":
    video1_path = "./output/output_superpoint_glue_matches.mp4"
    video2_path = "./output/output_superpoint_glue_visualization.mp4"
    output_path = "./output/combined_videos.mp4"
    
    # Combine videos side by side (horizontal)
    combine_videos(video1_path, video2_path, output_path, side_by_side=True)
    
    # Uncomment the following line to combine videos vertically instead
    # combine_videos(video1_path, video2_path, output_path, side_by_side=False)
