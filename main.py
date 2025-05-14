import cv2
import mediapipe as mp
from ultralytics import YOLO
import os

# === CONFIGS === #
USE_MEDIAPIPE = True
USE_YOLO = True

# === INIT MODELS === #
def init_models():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) if USE_MEDIAPIPE else None
    yolo_model = YOLO("yolov8n.pt") if USE_YOLO else None
    return mp_pose, mp_draw, pose, yolo_model

# === DRAW EXTRA CONNECTIONS === #
def draw_extra_connections(frame, landmarks, frame_width, frame_height):
    extra_connections = [
        (0, 1), (0, 4), (9, 10), (11, 13), (12, 14),
        (13, 15), (14, 16), (23, 24), (11, 23), (12, 24),
        (23, 25), (24, 26), (25, 27), (26, 28), (27, 31), (28, 32)
    ]
    for p1, p2 in extra_connections:
        x1, y1 = int(landmarks[p1].x * frame_width), int(landmarks[p1].y * frame_height)
        x2, y2 = int(landmarks[p2].x * frame_width), int(landmarks[p2].y * frame_height)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

# === PROCESS A SINGLE FRAME === #
def process_frame(frame, mp_pose, mp_draw, pose, yolo_model, width, height):
    # YOLO
    if yolo_model:
        yolo_results = yolo_model(frame, verbose=False)
        frame = yolo_results[0].plot()

    # MediaPipe Pose
    if pose:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            connection_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )
            draw_extra_connections(frame, results.pose_landmarks.landmark, width, height)
    return frame

# === MAIN PIPELINE === #
def run_pipeline(input_path, output_path):
    mp_pose, mp_draw, pose, yolo_model = init_models()
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"🎥 Processing {frame_count} frames from: {input_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, mp_pose, mp_draw, pose, yolo_model, width, height)
        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"✅ Saved processed video to: {output_path}")

# === ENTRY POINT === #
if __name__ == "__main__":
    input_video = input("📁 Enter input video path: ").strip()
    output_video = "yolo_output.mp4"
    run_pipeline(input_video, output_video)
