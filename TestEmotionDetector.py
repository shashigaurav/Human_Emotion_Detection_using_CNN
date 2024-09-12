import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the trained emotion detection model
emotion_model = Sequential()
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.2))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.22))

emotion_model.add(Flatten())
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(256, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions

emotion_model.load_weights("model/emotion_model.h5")

print("Loaded model weights from disk")


# Function to detect emotion for each frame
def detect_emotion(frame):
    # Find haar cascade to draw bounding box around face
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 163, 250), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        max_probability = emotion_prediction[0][max_index] * 100  # Get the probability of the predicted emotion

        # Print the emotion label and confidence score
        emotion_label = emotion_dict[max_index]
        print(f"Predicted Emotion: {emotion_label}, Confidence: {max_probability:.2f}%")

        # Draw the emotion label and confidence score on the frame
        percentage = f"{max_probability:.2f}%"
        cv2.putText(frame, emotion_label, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (89, 213, 224), 2, cv2.LINE_AA)
        cv2.putText(frame, percentage, (x+w-60, y-60), cv2.FONT_HERSHEY_SIMPLEX, .6, (138, 83, 224), 2, cv2.LINE_AA)





# Your input source code remains the same
input_source = 'camera'  # Change this to 'image', 'video', or 'camera' as needed

if input_source == 'image':
    input_file = 'test.jpg'  # Change this to your input file path
    frame = cv2.imread(input_file)
    detect_emotion(frame)
    cv2.imshow('Emotion Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif input_source == 'video':
    input_file = 'test.mp4'  # Change this to your input video file path
    cap = cv2.VideoCapture(input_file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_emotion(frame)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif input_source == 'camera':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_emotion(frame)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid input source.")