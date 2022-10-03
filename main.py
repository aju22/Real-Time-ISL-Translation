import cv2
import numpy as np
import os
import mediapipe as mp
from utils import mediapipe_detection, landmarks_data, prob_viz
from models import load_model


if __name__ == "__main__":
    sequence = []
    sentence = []
    predictions = []
    frame_count = 0
    res = None
    thresh = 0.85

    mp_holistic = mp.solutions.holistic
    model = load_model('lstm_v3', pretrained=True, training=False)
    actions = os.listdir('greetings_data')
    cap = cv2.VideoCapture('Test_video.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()
            frame_count += 1

            if ret and frame_count % 2 == 0:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                #draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = landmarks_data(results)
                sequence.append(keypoints)

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    sequence = []

                    if res[np.argmax(res)] > thresh:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image)

                cv2.rectangle(image, (0, 0), (width, 40), (0, 0, 0), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            elif ret is False:
                break
