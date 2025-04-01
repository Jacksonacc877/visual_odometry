import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import csv

def preprocess_dark_image(img):
    """Preprocess dark images to improve feature detection."""
    brightness = np.mean(img)
    
    if brightness < 100:
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        if brightness < 50:
            gamma = 1.5
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv.LUT(enhanced, lookup_table)
        
        return enhanced
    
    return img

def alternative_preprocess(img):
    """Alternative preprocessing method for cases where standard preprocessing fails."""
    filtered = cv.bilateralFilter(img, 9, 75, 75)
    thresh = cv.adaptiveThreshold(
        filtered, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY, 11, 2
    )
    result = cv.addWeighted(img, 0.7, thresh, 0.3, 0)
    return result

def init_kalman_filter():
    """Initialize Kalman filter for trajectory smoothing."""
    kf = cv.KalmanFilter(6, 3)
    
    kf.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], np.float32)
    
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], np.float32)
    
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    
    return kf

def update_kalman_filter(kf, position):
    """Update the Kalman filter with a new position measurement."""
    predicted = kf.predict()
    measurement = np.array(position, dtype=np.float32).reshape(3, 1)
    corrected = kf.correct(measurement)
    return corrected[:3].flatten()

def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to quaternion."""
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

def save_trajectory_data(raw_trajectory, smoothed_trajectory):
    """Save both raw and smoothed trajectory data to files."""
    trajectory_data = {
        'raw_trajectory': raw_trajectory,
        'smoothed_trajectory': smoothed_trajectory
    }
    np.save('./output/trajectory_data.npy', trajectory_data)
    
    for traj_type, trajectory in [('raw', raw_trajectory), ('smoothed', smoothed_trajectory)]:
        with open(f'./output/{traj_type}_trajectory.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
            
            for i, (R, t) in enumerate(trajectory):
                qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
                writer.writerow([i, t[0, 0], t[1, 0], t[2, 0], qw, qx, qy, qz])
    
    print("=> Trajectory data saved to trajectory_data.npy")
    print("=> Raw trajectory saved to raw_trajectory.csv")
    print("=> Smoothed trajectory saved to smoothed_trajectory.csv")

def render_trajectory(raw_trajectory, smoothed_trajectory, width, height):
    """Render both raw and smoothed camera trajectories using matplotlib."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract trajectory points
    raw_points = np.array([[t[0, 0], t[1, 0], t[2, 0]] for _, t in raw_trajectory])
    smooth_points = np.array([[t[0, 0], t[1, 0], t[2, 0]] for _, t in smoothed_trajectory])
    
    # Plot trajectories
    ax.plot(raw_points[:, 0], raw_points[:, 1], raw_points[:, 2], 'b-', linewidth=1, alpha=0.5, label='Raw')
    ax.plot(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], 'r-', linewidth=2, label='Smoothed')
    ax.plot([smooth_points[-1, 0]], [smooth_points[-1, 1]], [smooth_points[-1, 2]], 'ro')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend(loc='upper right', fontsize='small')
    
    # Set consistent axis limits
    for points in [raw_points, smooth_points]:
        ax.set_xlim([min(points[:, 0])-1, max(points[:, 0])+1])
        ax.set_ylim([min(points[:, 1])-1, max(points[:, 1])+1])
        ax.set_zlim([min(points[:, 2])-1, max(points[:, 2])+1])
    
    # Convert plot to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert to BGR and resize
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    if img.shape[:2] != (height, width):
        img = cv.resize(img, (width, height))
    
    return img
