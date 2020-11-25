# To capture face recognition using a VIDEO file.
import os 
import cv2
import face_recognition

# Import your video file
video_file = cv2.VideoCapture(0)

# We need to add all the faces that we want our code to recognize
image_monnish = face_recognition.load_image_file(os.path.abspath("recognize/images/monnish.jpg"))
image_lohit = face_recognition.load_image_file(os.path.abspath("recognize/images/lohit.jpeg"))
image_logesh = face_recognition.load_image_file(os.path.abspath("recognize/images/logesh.jpeg"))

# Generate the face encoding for the image that has been passed.
monnish_face = face_recognition.face_encodings(image_monnish)[0]
lohit_face = face_recognition.face_encodings(image_lohit)[0]
logesh_face = face_recognition.face_encodings(image_logesh)[0]

# Make a list of all the known faces that we want to be recognized based on the 
# encoding.
known_faces = [
monnish_face,lohit_face,logesh_face
]

facial_points = []
face_encodings = []

while video_file.isOpened():
    return_value, frame = video_file.read()
    
    if not return_value:
        break
    rgb_frame = frame[:, :, ::-1]

    facial_points = face_recognition.face_locations(rgb_frame, model="hog")
    #print(facial_points)
    face_encodings = face_recognition.face_encodings(rgb_frame, facial_points)
    #print(face_recognition.face_encodings(rgb_frame, facial_points))
    #print(face_encodings)
    facial_names = []
    for encoding in face_encodings:
        #print(encoding)
        match = face_recognition.compare_faces(known_faces, encoding, tolerance=0.50)
        # monnish_face,lohit_face,logesh_face
        # match = [True, False , False]

        name = ""
        if match[0]:
            name = "monnish"
        if match[1]:
            name = "lohit"
        if match[2]:
            name = "logesh"

        facial_names.append(name)
        #zip(facial_points, facial_names)

    for (top, right, bottom, left),name in zip(facial_points, facial_names):
        # Enclose the face with the box - Red color 
        # top, right, bottom, left - 129, 710, 373, 465
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Name the characters in the Box created above
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    cv2.putText(frame,'PRESS q TO STOP SCAN', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video_file.release()
cv2.destroyAllWindows()
