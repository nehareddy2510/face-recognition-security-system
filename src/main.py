import dlib
import os
import cv2
import numpy as np

detector = dlib.get_frontal_face_detector() 
sp = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

facerecognition = dlib.face_recognition_model_v1(
   "dlib_face_recognition_resnet_model_v1.dat"
)


def encoding_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue  

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                faces = detector(rgb_image)

                for face in faces:
                    # Get the landmarks
                    shape = sp(rgb_image, face)

                    # Get the face encoding (128-d vector)
                    encoding = facerecognition.compute_face_descriptor(rgb_image, shape)

                    # Convert the encoding to a list
                    encoding = np.array(encoding)

                    # Add the encoding and name to the lists
                    known_face_encodings.append(encoding)
                    known_face_names.append(person_name)
    
    return known_face_encodings, known_face_names

dataset_path = "dataset/faces"

known_face_encodings, known_face_names =encoding_faces(dataset_path)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("failed to open video capture")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  
cv2.resizeWindow("Frame", 700, 700)  
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
 

while True:
        
    ret, frame = cap.read()
    if not ret:
        break
    #for dlib rgb
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_frame)

    for face in faces:
        shape = sp(rgb_frame, face)
        encoding = facerecognition.compute_face_descriptor(rgb_frame, shape)
        encoding = np.array(encoding)

        # Compare with known faces
        distances = np.linalg.norm(known_face_encodings - encoding, axis=1)
        threshold = 0.6
        matches = distances < threshold
        name = "Intruder"

        if True in matches:
            first_match_index = matches.argmax()
            name = known_face_names[first_match_index]

        (top, right, bottom, left) = (face.top(), face.right(), face.bottom(), face.left())
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 3)
    
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
