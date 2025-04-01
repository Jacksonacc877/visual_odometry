import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import matplotlib.pyplot as plt
import csv
import torch

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
        self.superglue= args.superglue
        # Initialize SuperGlue if available
        if self.superglue:
            try:
                self.init_superglue()
                print("=> SuperGlue and SuperPoint initialized successfully")
            except Exception as e:
                print(f"=> SuperGlue/SuperPoint initialization failed: {e}")

    def init_superglue(self):
        """Initialize SuperGlue matcher with SuperPoint detector."""
        # Configuration for SuperPoint and SuperGlue
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'outdoor',  # 'outdoor' for outdoor scenes
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        
        # Initialize the matching model (includes both SuperPoint and SuperGlue)
        self.matcher = Matching(config).eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.matcher = self.matcher.to('cuda')
            self.device = 'cuda'
            print("=> Using GPU for SuperPoint/SuperGlue")
        else:
            self.device = 'cpu'
            print("=> Using CPU for SuperPoint/SuperGlue")

    def run(self):
        # Initialize video writers with different paths for SuperPoint/SuperGlue
        output_video_path = "./output/output_superpoint_glue_visualization.mp4"
        matches_video_path = "./output/output_superpoint_glue_matches.mp4"
        frame_width, frame_height = 640, 480  # Set desired resolution
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))
        matches_writer = cv.VideoWriter(matches_video_path, fourcc, 30, (frame_width, frame_height))
        
        # For tracking trajectory
        raw_trajectory = []
        
        # Process frames in the main process to avoid multiprocessing issues with Open3D
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        raw_trajectory.append((R.copy(), t.copy()))  # Save initial position
        
        prev_img = cv.imread(self.frame_paths[0], cv.IMREAD_GRAYSCALE)
        
        # For SuperPoint/SuperGlue, we'll store images and process them together
        prev_color = cv.imread(self.frame_paths[0])
        
        # Initialize prevous keypoints and descriptors
        if self.superglue:
            # SuperPoint/SuperGlue will extract features during matching
            prev_keypoints, prev_descriptors = None, None
            prev_data = {'image': prev_img.copy(), 'color': prev_color.copy()}
        else:
            # Use traditional feature detection
            prev_keypoints, prev_descriptors = self.detect_and_compute(prev_img)

        # Initialize the Kalman filter for trajectory smoothing
        kf = self.init_kalman_filter()
        
        frame_counter = 0
        for frame_path in self.frame_paths[1:]:
            frame_counter += 1
            img = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
            curr_color = cv.imread(frame_path)
            
            if self.superglue:
                # Store current images for SuperGlue
                curr_data = {'image': img.copy(), 'color': curr_color.copy()}
                
                # Process together with SuperPoint/SuperGlue
                keypoints, descriptors, matches, prev_keypoints, prev_descriptors = self.process_with_superpoint_superglue(
                    prev_data, curr_data)
                
                # Update images for next iteration
                prev_data = curr_data
            else:
                # Traditional feature detection
                keypoints, descriptors = self.detect_and_compute(img)
                matches = self.match_features(prev_descriptors, descriptors, prev_img, img, prev_keypoints, keypoints)
            
            try:
                # Match features
                if len(matches) > 8:  # Minimum matches for essential matrix
                    pts1 = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

                    # Compute essential matrix
                    E, mask = cv.findEssentialMat(pts1, pts2, self.K, method=cv.RANSAC, prob=0.999, threshold=1.0)
                    _, R_new, t_new, _ = cv.recoverPose(E, pts1, pts2, self.K)

                    # Update pose
                    R = R @ R_new
                    t = t + R @ t_new
                    
                    # Save raw trajectory
                    raw_trajectory.append((R.copy(), t.copy()))
                    
                    # Update the Kalman filter with the new position
                    current_pos = np.array([t[0, 0], t[1, 0], t[2, 0]], dtype=np.float32)
                    smoothed_pos = self.update_kalman_filter(kf, current_pos)
                    
                    # Build smoothed trajectory for visualization
                    smoothed_trajectory = self.build_smoothed_trajectory(raw_trajectory, kf)
                    
                    # Create visualization using matplotlib (now with both raw and smoothed)
                    trajectory_frame = self.render_trajectory(raw_trajectory, smoothed_trajectory, frame_width, frame_height)
                    
                    # Create matches visualization
                    matches_frame = self.draw_matches(prev_color, prev_keypoints, curr_color, keypoints, matches, mask if 'mask' in locals() else None)
                    
                    # Write both frames to their respective videos
                    video_writer.write(trajectory_frame)
                    matches_writer.write(matches_frame)

                if self.superglue:
                    prev_keypoints, prev_descriptors = keypoints, descriptors
                    prev_img = img.copy()
                    prev_color = curr_color.copy()
                
                print(f"Processing frame {frame_counter}/{len(self.frame_paths)}", end='\r')
            except Exception as e:
                print(f"Error processing frame {frame_counter}: {e}")
            # cv.imshow('frame', img)
            # if cv.waitKey(30) == 27: break

        video_writer.release()
        matches_writer.release()
        print(f"=> Visualization saved to {output_video_path}")
        print(f"=> Matches visualization saved to {matches_video_path}")
        
        # Build final smoothed trajectory
        smoothed_trajectory = self.build_smoothed_trajectory(raw_trajectory, kf)
        
        # Save both raw and smoothed trajectory data
        self.save_trajectory_data(raw_trajectory, smoothed_trajectory)
        
        # Save final trajectory plot
        self.save_trajectory_plot(raw_trajectory, smoothed_trajectory)

    def process_with_superpoint_superglue(self, prev_data, curr_data):
        """Process a pair of images using SuperPoint and SuperGlue."""
        # Convert grayscale images to tensors
        tensor1 = frame2tensor(prev_data['image'], self.device)
        tensor2 = frame2tensor(curr_data['image'], self.device)
        
        # Prepare data for the matcher (contains both SuperPoint and SuperGlue)
        data = {
            'image0': tensor1,
            'image1': tensor2,
        }
        
        # Process data with SuperPoint + SuperGlue
        with torch.no_grad():
            pred = self.matcher(data)
        
        # Extract keypoints
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        
        # Create OpenCV keypoints
        cv_kpts0 = [cv.KeyPoint(float(x), float(y), 1) for x, y in kpts0]
        cv_kpts1 = [cv.KeyPoint(float(x), float(y), 1) for x, y in kpts1]
        
        # Extract matches
        matches0 = pred['matches0'][0].cpu().numpy()
        valid = matches0 > -1
        
        # Convert to OpenCV DMatch format
        cv_matches = []
        for i, m in enumerate(matches0):
            if m > -1:  # if point i in image 0 matches with point m in image 1
                confidence = pred['matching_scores0'][0][i].cpu().item()
                cv_matches.append(cv.DMatch(i, m, 1.0 - confidence))  # lower is better for OpenCV matches
        
        # Sort by confidence (ascending distance)
        cv_matches = sorted(cv_matches, key=lambda x: x.distance)
        
        # Create dummy descriptors for compatibility with existing code
        # (SuperGlue doesn't expose the descriptors it uses internally)
        dummy_desc0 = np.ones((len(cv_kpts0), 128), dtype=np.float32)
        dummy_desc1 = np.ones((len(cv_kpts1), 128), dtype=np.float32)
        
        return cv_kpts1, dummy_desc1, cv_matches, cv_kpts0, dummy_desc0

    def init_kalman_filter(self):
        """Initialize Kalman filter for trajectory smoothing."""
        kf = cv.KalmanFilter(6, 3)  # 6 states (position + velocity), 3 measurements (position)
        
        # State transition matrix (constant velocity model)
        kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1],  # z = z + vz
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1]   # vz = vz
        ], np.float32)
        
        # Measurement matrix (we only measure position, not velocity)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
        # Error covariance
        kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        return kf
        
    def update_kalman_filter(self, kf, position):
        """Update the Kalman filter with a new position measurement."""
        # Kalman predict
        predicted = kf.predict()
        
        # Kalman update
        measurement = np.array(position, dtype=np.float32).reshape(3, 1)
        corrected = kf.correct(measurement)
        
        # Return the smoothed position (first 3 elements of state)
        return corrected[:3].flatten()
        
    def build_smoothed_trajectory(self, raw_trajectory, kf):
        """Build a complete smoothed trajectory from the raw trajectory using Kalman filtering."""
        smoothed_trajectory = []
        
        # Reset the Kalman filter
        kf = self.init_kalman_filter()
        
        # Initialize with the first position
        R_init, t_init = raw_trajectory[0]
        smoothed_pos = np.array([t_init[0, 0], t_init[1, 0], t_init[2, 0]], dtype=np.float32)
        smoothed_trajectory.append((R_init.copy(), t_init.copy()))
        
        # Apply Kalman filter to each position
        for i in range(1, len(raw_trajectory)):
            R, t = raw_trajectory[i]
            current_pos = np.array([t[0, 0], t[1, 0], t[2, 0]], dtype=np.float32)
            
            # Update filter and get smoothed position
            smoothed_pos = self.update_kalman_filter(kf, current_pos)
            
            # Create a new translation vector with the smoothed position
            t_smoothed = np.array([[smoothed_pos[0]], [smoothed_pos[1]], [smoothed_pos[2]]], dtype=np.float64)
            
            # Keep the same rotation matrix
            smoothed_trajectory.append((R.copy(), t_smoothed))
        
        return smoothed_trajectory

    def draw_matches(self, img1, kp1, img2, kp2, matches, mask=None):
        """Draw the matches between two images."""
        # Create a blank canvas with the same dimensions as our output
        height, width = 480, 640
        
        # Resize input images if needed
        img1 = cv.resize(img1, (width//2, height))
        img2 = cv.resize(img2, (width//2, height))
        
        # Create a side-by-side image
        matches_img = np.zeros((height, width, 3), dtype=np.uint8)
        matches_img[:, :width//2] = img1
        matches_img[:, width//2:] = img2
        
        # Use different colors for SuperPoint matches to distinguish them
        color = (0, 255, 0)  # Default green for traditional matches
        if self.superglue:
            color = (0, 165, 255)  # Orange for SuperPoint+SuperGlue matches
        
        # If mask is available, use it to filter matches
        good_matches = []
        if mask is not None:
            mask = mask.ravel().tolist()
            good_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        else:
            # Use all matches or limit to a reasonable number to avoid cluttering
            good_matches = matches[:50] if len(matches) > 50 else matches
        
        # Draw lines between the matches
        for match in good_matches:
            # Get the keypoints for each match
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            
            # Get keypoint coordinates
            (x1, y1) = map(int, kp1[img1_idx].pt)
            (x2, y2) = map(int, kp2[img2_idx].pt)
            
            # Adjust x2 by adding width//2 to account for side-by-side layout
            x2 += width // 2
            
            # Draw circles around the keypoints
            cv.circle(matches_img, (x1, y1), 4, color, 1)
            cv.circle(matches_img, (x2, y2), 4, color, 1)
            
            # Draw a line connecting the keypoints
            cv.line(matches_img, (x1, y1), (x2, y2), color, 1)
        
        # Add a label to indicate which method was used
        method_label = "SuperPoint+SuperGlue" if self.superglue else "Traditional Features"
        cv.putText(matches_img, method_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(matches_img, f"Matches: {len(good_matches)}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return matches_img

    def detect_and_compute(self, img):
        """Detect keypoints and compute descriptors using more robust methods.
        Includes preprocessing for dark frames."""
        # Preprocess the image to enhance features in dark areas
        preprocessed_img = self.preprocess_dark_image(img)
        
        # If we're using SuperGlue, we'll only detect keypoints here
        # SuperGlue will handle the descriptors and matching later
        if self.superglue:
            try:
                # Use SIFT for keypoint detection (SuperPoint will be used by SuperGlue)
                sift = cv.SIFT_create(nfeatures=2000)
                keypoints, descriptors = sift.detectAndCompute(preprocessed_img, None)
                
                # For SuperGlue compatibility later on
                self.last_processed_img = preprocessed_img.copy()
                
                return keypoints, descriptors
            except:
                pass  # Fall through to standard approach if SIFT fails
        
        # If SuperGlue is not available or failed, use the existing approach
        try:
            # Try SIFT first (better for low-light conditions)
            sift = cv.SIFT_create(nfeatures=2000)
            keypoints, descriptors = sift.detectAndCompute(preprocessed_img, None)
        except:
            try:
                # Fallback to AKAZE if SIFT isn't available
                akaze = cv.AKAZE_create()
                keypoints, descriptors = akaze.detectAndCompute(preprocessed_img, None)
            except:
                # Final fallback to ORB with adjusted parameters
                orb = cv.ORB_create(nfeatures=3000, 
                                    scaleFactor=1.1,
                                    edgeThreshold=15,
                                    patchSize=31,
                                    fastThreshold=5)
                keypoints, descriptors = orb.detectAndCompute(preprocessed_img, None)
        
        # If we still don't have enough keypoints, try a different preprocessing
        if keypoints is None or len(keypoints) < 50:
            # Try alternative preprocessing
            alt_img = self.alternative_preprocess(img)
            
            try:
                # Try SIFT with alternative preprocessing
                sift = cv.SIFT_create(nfeatures=2000)
                keypoints, descriptors = sift.detectAndCompute(alt_img, None)
            except:
                # Fallback to ORB with adjusted parameters
                orb = cv.ORB_create(nfeatures=3000)
                keypoints, descriptors = orb.detectAndCompute(alt_img, None)
                
        if descriptors is None:
            # If still no descriptors, create empty arrays to avoid errors
            keypoints = []
            descriptors = np.array([])
            
        return keypoints, descriptors

    def preprocess_dark_image(self, img):
        """Preprocess dark images to improve feature detection."""
        # Check if image is too dark
        brightness = np.mean(img)
        
        if brightness < 100:  # Threshold for "dark" images
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img)
            
            # Apply additional brightness correction if needed
            if brightness < 50:  # Very dark images
                # Gamma correction
                gamma = 1.5
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv.LUT(enhanced, lookup_table)
            
            return enhanced
        
        return img
    
    def alternative_preprocess(self, img):
        """Alternative preprocessing method for cases where standard preprocessing fails."""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv.bilateralFilter(img, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv.adaptiveThreshold(
            filtered, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY, 11, 2
        )
        
        # Combine original with thresholded image
        result = cv.addWeighted(img, 0.7, thresh, 0.3, 0)
        
        return result

    def match_features_superglue(self, img1, kp1, img2, kp2):
        """This method is kept for backward compatibility but should not be used 
        when using complete SuperPoint+SuperGlue pipeline."""
        print("Warning: match_features_superglue called directly, which is not optimal with SuperPoint. " 
              "Use process_with_superpoint_superglue instead.")
        
        if not self.superglue:
            return []
            
        try:
            # Convert images to tensors
            tensor1 = frame2tensor(img1, self.device)
            tensor2 = frame2tensor(img2, self.device)
            
            # Extract keypoints as tensors
            kpts1 = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kp1], 
                                 dtype=torch.float32, device=self.device)
            kpts2 = torch.tensor([[kp.pt[0], kp.pt[1]] for kp in kp2], 
                                 dtype=torch.float32, device=self.device)
            
            # Prepare data for SuperGlue
            data = {
                'image0': tensor1,
                'image1': tensor2,
                'keypoints0': kpts1.unsqueeze(0),
                'keypoints1': kpts2.unsqueeze(0),
                'scores0': torch.ones(kpts1.shape[0], device=self.device).unsqueeze(0),
                'scores1': torch.ones(kpts2.shape[0], device=self.device).unsqueeze(0),
            }
            
            # Process with SuperGlue
            with torch.no_grad():
                pred = self.superglue(data)
            
            # Get matches
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()
            
            # Convert to OpenCV DMatch format
            cv_matches = []
            for i, m in enumerate(matches):
                if m != -1 and confidence[i] > 0.5:  # Only use matches above threshold
                    cv_matches.append(cv.DMatch(i, m, 1-confidence[i]))  # Lower distance is better
            
            # Sort matches by confidence (lowest distance first)
            cv_matches = sorted(cv_matches, key=lambda x: x.distance)
            
            return cv_matches
        except Exception as e:
            print(f"SuperGlue matching failed: {e}")
            return []

    def match_features(self, desc1, desc2, prev_img=None, curr_img=None, prev_kp=None, curr_kp=None):
        """Match features using SuperGlue if available, otherwise fall back to conventional methods."""
        # Try SuperGlue first if images and keypoints are provided
        if self.superglue and prev_img is not None and curr_img is not None and prev_kp is not None and curr_kp is not None:
            sg_matches = self.match_features_superglue(prev_img, prev_kp, curr_img, curr_kp)
            if len(sg_matches) > 10:  # If we got enough matches from SuperGlue
                print(f"Using SuperGlue matches: {len(sg_matches)}")
                return sg_matches
        
        # Fall back to conventional matching
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
            
        # Check descriptor type and use appropriate matcher
        if desc1.dtype == np.float32 and desc2.dtype == np.float32:
            # SIFT/SURF descriptors are float32 and use NORM_L2
            matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        else:
            # ORB/BRIEF/BRISK descriptors are binary and use NORM_HAMMING
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            
        # Get matches
        matches = matcher.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Only use good matches
        good_match_ratio = 0.75
        num_good_matches = int(len(matches) * good_match_ratio)
        return matches[:num_good_matches] if num_good_matches > 0 else matches

    def render_trajectory(self, raw_trajectory, smoothed_trajectory, width, height):
        """Render both raw and smoothed camera trajectories using matplotlib."""
        # Create a figure with a 3D subplot
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract raw trajectory points
        raw_x = []
        raw_y = []
        raw_z = []
        
        for R, t in raw_trajectory:
            raw_x.append(t[0, 0])
            raw_y.append(t[1, 0])
            raw_z.append(t[2, 0])
        
        # Extract smoothed trajectory points
        smooth_x = []
        smooth_y = []
        smooth_z = []
        
        for R, t in smoothed_trajectory:
            smooth_x.append(t[0, 0])
            smooth_y.append(t[1, 0])
            smooth_z.append(t[2, 0])
        
        # Plot the raw trajectory (thin blue line)
        ax.plot(raw_x, raw_y, raw_z, 'b-', linewidth=1, alpha=0.5, label='Raw')
        
        # Plot the smoothed trajectory (thicker red line)
        ax.plot(smooth_x, smooth_y, smooth_z, 'r-', linewidth=2, label='Smoothed')
        
        # Mark the current position
        ax.plot([smooth_x[-1]], [smooth_y[-1]], [smooth_z[-1]], 'ro')
        
        # Set axis labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        ax.legend(loc='upper right', fontsize='small')
        
        # If needed for consistent visualization:
        ax.set_xlim([min(min(raw_x), min(smooth_x))-1, max(max(raw_x), max(smooth_x))+1])
        ax.set_ylim([min(min(raw_y), min(smooth_y))-1, max(max(raw_y), max(smooth_y))+1])
        ax.set_zlim([min(min(raw_z), min(smooth_z))-1, max(max(raw_z), max(smooth_z))+1])
        
        # Convert plot to an image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        # Resize if needed to match video dimensions
        if img.shape[0] != height or img.shape[1] != width:
            img = cv.resize(img, (width, height))
            
        return img
    
    def save_trajectory_plot(self, raw_trajectory, smoothed_trajectory):
        """Save high quality trajectory plots showing both raw and smoothed trajectories."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract raw trajectory points
        raw_x = []
        raw_y = []
        raw_z = []
        
        for R, t in raw_trajectory:
            raw_x.append(t[0, 0])
            raw_y.append(t[1, 0])
            raw_z.append(t[2, 0])
        
        # Extract smoothed trajectory points
        smooth_x = []
        smooth_y = []
        smooth_z = []
        
        for R, t in smoothed_trajectory:
            smooth_x.append(t[0, 0])
            smooth_y.append(t[1, 0])
            smooth_z.append(t[2, 0])
        
        # Plot the raw trajectory
        ax.plot(raw_x, raw_y, raw_z, 'b-', linewidth=1, alpha=0.5, label='Raw Trajectory')
        
        # Plot the smoothed trajectory
        ax.plot(smooth_x, smooth_y, smooth_z, 'r-', linewidth=2, label='Smoothed Trajectory')
        
        # Mark start and end positions
        ax.plot([raw_x[0]], [raw_y[0]], [raw_z[0]], 'go', markersize=8, label='Start')
        ax.plot([smooth_x[-1]], [smooth_y[-1]], [smooth_z[-1]], 'ro', markersize=8, label='End')
        
        # Set axis labels, title and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory Comparison')
        ax.legend()
        
        # Save the plot
        plt.savefig('./output/trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("=> Trajectory comparison plot saved to trajectory_comparison.png")
    
    def save_trajectory_data(self, raw_trajectory, smoothed_trajectory):
        """Save both raw and smoothed trajectory data to files."""
        # Save as NumPy file
        trajectory_data = {
            'raw_trajectory': raw_trajectory,
            'smoothed_trajectory': smoothed_trajectory
        }
        np.save('./output/trajectory_data.npy', trajectory_data)
        
        # Save as CSV for easy import into other tools
        with open('./output/raw_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            
            for i, (R, t) in enumerate(raw_trajectory):
                # Convert rotation matrix to quaternion
                qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
                writer.writerow([i, t[0, 0], t[1, 0], t[2, 0], qw, qx, qy, qz])
        
        with open('./output/smoothed_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            
            for i, (R, t) in enumerate(smoothed_trajectory):
                # Convert rotation matrix to quaternion
                qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
                writer.writerow([i, t[0, 0], t[1, 0], t[2, 0], qw, qx, qy, qz])
        
        print("=> Trajectory data saved to trajectory_data.npy")
        print("=> Raw trajectory saved to raw_trajectory.csv")
        print("=> Smoothed trajectory saved to smoothed_trajectory.csv")
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert a rotation matrix to quaternion."""
        # Adapted from transforms3d or similar libraries
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        return qw, qx, qy, qz

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='./camera_data/camera_parameters.npy', help='npy file of camera parameters')
    parser.add_argument('--superglue', default=True, help='use SuperGlue for feature matching')

    args = parser.parse_args()
    print("torch cuda available:", torch.cuda.is_available())
    
    if not os.path.exists("./output"):
        os.makedirs("./output")

    vo = SimpleVO(args)
    vo.run()
