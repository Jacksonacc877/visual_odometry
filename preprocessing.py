import cv2 as cv
import os

def preprocess_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("=> End of video reached.")
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv.imwrite(frame_path, frame)
        print(f"Saved frame: {frame_path}")
        frame_count += 1

    cap.release()
    print(f"=> Total frames saved: {frame_count}")

if __name__ == "__main__":
    video_path = "test_video.mp4"
    output_dir = "new_frames"
    preprocess_video(video_path, output_dir)
