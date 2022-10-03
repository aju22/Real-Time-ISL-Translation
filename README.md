# Real-Time-ISL-Translation

## ***Introduction***

Indian Sign Language is not only a means of communication for the hearing impaired, but is a symbol of pride and idendity.
Strenuous efforts have been made by Deaf communities, NGO's, researchers and other organisations working for people with hearing disabilities , including the All India Federation of Deaf (AIFD), National association of the Deaf (NAD) in the direction of encouraging ISL.

There has been some significant amount of research on Sign language translation, but with very less focus for Indo sign language.

This project presents a system which can recognise gestures from the Indian Sign Language (ISL) using ***Mediapipe Pose Detection Library*** and the feeding the data points through an ***LSTM Network***, enabling real-time prediction of the language. This attempts to bridge the communication gap between the hearing and speech impaired and the rest of the society.

## ***Architecture***

<ul>
  <li>Feed the video sequence to MediaPipe Pose Detection Library.</li>
  <li>Extract Arms and Body Posture keypoints.</li>
  <li>Feed the sequence data into an LSTM Network.</li>
  <li>Predict classes of gestures.</li>
</ul>


