import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    # Get frame from camera
    ret, frame = cap.read()

    # Detect and track faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    # Cartoonize each face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]

        # Convert the face ROI to a grayscale image
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Apply the stylization effect
        stylized_face = cv2.stylization(gray_face, sigma_s=60, sigma_r=0.07)

        # Convert the stylized face back to the original color format
        stylized_face = cv2.cvtColor(stylized_face, cv2.COLOR_GRAY2BGR)

        # Overlay the stylized face ROI back on the original frame
        frame[y:y+h, x:x+w] = stylized_face

    # Display the cartoonized frame
    cv2.imshow("Cartoonized Video", frame)
