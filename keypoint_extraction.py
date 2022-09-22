import os
import cv2
import mediapipe as mp
import numpy as np
from utils import mediapipe_detection, landmarks_data, pad_sequence
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def save_data(action, video_file, export_path, import_path, max_frame_length=30, skip_frame=2):
    MAX_FRAME_LENGTH = max_frame_length
    EXPORT_PATH = export_path
    IMPORT_PATH = import_path
    SKIP_FRAME = skip_frame

    frame_count = 0
    processed = 0
    data_per_video = []

    cap = cv2.VideoCapture(os.path.join(IMPORT_PATH, action, video_file))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            frame_count += 1

            if ret and (frame_count % SKIP_FRAME == 0):

                image, results = mediapipe_detection(frame, holistic)
                processed += 1

                data_per_video.append(landmarks_data(results))

            elif ret is False or processed == MAX_FRAME_LENGTH:

                if processed != MAX_FRAME_LENGTH:
                    data_per_video = np.array(pad_sequence(data_per_video, MAX_FRAME_LENGTH))
                else:
                    data_per_video = np.array(data_per_video)
                os.makedirs(os.path.join(EXPORT_PATH, action), exist_ok=True)
                npy_path = os.path.join(EXPORT_PATH, action, f"{video_file[:-4]}_skip_{skip_frame}")
                print(f' Current Action : {action}\n Current Video : {video_file}\n',
                      f'Frames Processed : {processed}\n Data Shape : {data_per_video.shape}\n',
                      f'Skipped Frames : {skip_frame}\n',
                      f'Saving data to : {npy_path}\n',
                      "---------------------------------------------\n"
                      )
                np.save(npy_path, data_per_video)
                break

        cap.release()


if __name__ == "__main__":

    # Path for exported data, numpy arrays
    EXPORT_PATH = os.path.join('keypoint_data')
    IMPORT_PATH = os.path.join('greetings_data')
    MAX_FRAME_LENGTH = 30

    # Actions that we try to detect
    actions = os.listdir(IMPORT_PATH)

    for action in actions:

        for video_file in os.listdir(os.path.join(IMPORT_PATH, action)):

            save_data(action, video_file, EXPORT_PATH, IMPORT_PATH, MAX_FRAME_LENGTH, skip_frame=2)
        # save_data(action, video_file, EXPORT_PATH, IMPORT_PATH, MAX_FRAME_LENGTH, skip_frame=3)

