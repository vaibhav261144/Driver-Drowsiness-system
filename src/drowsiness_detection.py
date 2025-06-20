import cv2
import numpy as np
import time
import os
import argparse

# Constants for drowsiness detection
EYE_AR_CONSEC_FRAMES = 10  # Reduced threshold for faster detection

def detect_drowsiness():
    # Load OpenCV face detector and eye detector
    print("Loading face and eye detectors...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')  # Better eye detector
    
    # Initialize variables
    frame_counter = 0
    alarm_on = False
    
    # Start webcam feed
    print("Starting video stream...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    time.sleep(1.0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast of grayscale image
        gray = cv2.equalizeHist(gray)
        
        # Detect faces - more sensitive parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))
        
        # Debug info
        face_count = len(faces)
        status = "No faces detected"
        color = (0, 0, 255)  # Red for no faces
        
        if face_count > 0:
            status = f"Detected {face_count} face(s)"
            color = (0, 255, 0)  # Green for faces found
        
        # Display status text
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region - more sensitive parameters
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
            
            # Debug info
            eye_count = len(eyes)
            eye_status = f"Eyes: {eye_count} detected"
            cv2.putText(frame, eye_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # If no eyes are detected, increment counter
            if len(eyes) == 0:
                frame_counter += 1
                
                # If eyes closed for enough frames, trigger alarm
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    # If alarm not on, turn it on
                    if not alarm_on:
                        alarm_on = True
                        
                    # Draw alarm on frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display frame counter
                    cv2.putText(frame, f"Eyes closed for {frame_counter} frames", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Reset counter if eyes are detected
                frame_counter = 0
                alarm_on = False
                
                # Draw rectangles around eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Display frame counter if no faces detected but counter is active
        if face_count == 0 and frame_counter > 0:
            cv2.putText(frame, f"No detection for {frame_counter} frames", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Increment counter for no face detection
            frame_counter += 1
            
            # Trigger alarm if no face detected for too long
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alarm_on = True
        
        # Show frame
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (frame.shape[1]-150, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Check for key press (q to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_drowsiness() 