import numpy as np
import cv2
import face_recognition
import glob
import os
import time


images = glob.glob('data/*.jpg')

known_image_encodings = []
known_image_names = []
for image_path in images:
    # Load a sample picture and learn how to recognize it.
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(gray)
    image_encoding = face_recognition.face_encodings(image, face_locations)[0]
    image_name = image_path.split(os.path.sep)[-1].split(".")[0]
    known_image_encodings.append(image_encoding)
    known_image_names.append(image_name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

test_paths = glob.glob('test/*.jpg')

for path in test_paths:
    t = time.time()

    image_test = cv2.imread(path)

    gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_image_test = cv2.resize(image_test, (0, 0), fx=0.25, fy=0.25)
    small_image_test_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_image_test_gray)
    face_encodings = face_recognition.face_encodings(small_image_test, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_image_encodings, face_encoding, 0.4)
        name = "unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_image_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_image_names[best_match_index]

        face_names.append(name)



    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the image_test we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # Draw a box around the face
        cv2.rectangle(image_test, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image_test, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_test, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # cv2.imshow('Video', image_test)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    print("FPS: %.2f" % (1/(time.time() - t)))

    cv2.imwrite("results/%s" % path, image_test)
# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
