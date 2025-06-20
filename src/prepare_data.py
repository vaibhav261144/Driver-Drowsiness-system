import cv2
import dlib
import numpy as np
import os
import argparse
from imutils import face_utils
import time

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Eye Dataset for Training")
    parser.add_argument("--output_dir", type=str, default="../data",
                        help="Directory to save collected eye images")
    parser.add_argument("--shape_predictor", type=str, 
                        default="models/shape_predictor_68_face_landmarks.dat",
                        help="Path to facial landmark predictor")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to collect for each class")
    return parser.parse_args()

def setup_directories(output_dir):
    """
    Create the necessary directory structure for training and validation data
    """
    # Create main directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    
    # Create class subdirectories
    for main_dir in [train_dir, val_dir]:
        os.makedirs(os.path.join(main_dir, "open"), exist_ok=True)
        os.makedirs(os.path.join(main_dir, "closed"), exist_ok=True)
    
    return train_dir, val_dir

def extract_eyes(frame, shape, left_eye_idxs, right_eye_idxs):
    """
    Extract and preprocess eye regions from the frame
    """
    # Extract the left and right eye coordinates
    leftEye = shape[left_eye_idxs[0]:left_eye_idxs[1]]
    rightEye = shape[right_eye_idxs[0]:right_eye_idxs[1]]
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract eye regions
    left_eye_region = extract_eye_region(gray, leftEye)
    right_eye_region = extract_eye_region(gray, rightEye)
    
    return left_eye_region, right_eye_region

def extract_eye_region(gray, eye):
    """
    Extract the eye region from the grayscale frame
    """
    # Compute the bounding box of the eye
    (x, y, w, h) = cv2.boundingRect(eye)
    
    # Add some padding
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding
    
    # Extract the eye region
    eye_region = gray[y:y+h, x:x+w]
    
    # Resize to standard size
    eye_region = cv2.resize(eye_region, (24, 24))
    
    return eye_region

def collect_data():
    """
    Collect eye images for open and closed states
    """
    args = get_args()
    
    # Setup directories
    train_dir, val_dir = setup_directories(args.output_dir)
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    
    # Get the indexes of the facial landmarks for the left and right eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # Start webcam feed
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    # Counters
    open_count = 0
    closed_count = 0
    total_needed = args.num_samples
    
    # Collection state
    collecting = False
    current_class = "open"  # Start with open eyes
    
    print("Press 's' to start/stop collecting data")
    print("Press 'o' to switch to collecting 'open' eye images")
    print("Press 'c' to switch to collecting 'closed' eye images")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy for display
        display = frame.copy()
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        
        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye regions
            left_eye, right_eye = extract_eyes(frame, shape, (lStart, lEnd), (rStart, rEnd))
            
            # Draw rectangles around the eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(display, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # If collecting, save the eye images
            if collecting:
                # Determine where to save (80% train, 20% validation)
                is_train = np.random.rand() < 0.8
                save_dir = os.path.join(train_dir if is_train else val_dir, current_class)
                
                # Save left eye
                left_filename = os.path.join(save_dir, f"{current_class}_left_{time.time()}.png")
                cv2.imwrite(left_filename, left_eye)
                
                # Save right eye
                right_filename = os.path.join(save_dir, f"{current_class}_right_{time.time()}.png")
                cv2.imwrite(right_filename, right_eye)
                
                # Update counter
                if current_class == "open":
                    open_count += 2  # +2 because we save both eyes
                else:
                    closed_count += 2
                
                # Check if we have enough samples
                if current_class == "open" and open_count >= total_needed:
                    print(f"Collected enough 'open' eye samples ({open_count})")
                    collecting = False
                elif current_class == "closed" and closed_count >= total_needed:
                    print(f"Collected enough 'closed' eye samples ({closed_count})")
                    collecting = False
        
        # Display status
        status_text = f"Collecting: {'Yes' if collecting else 'No'} | Class: {current_class}"
        count_text = f"Open: {open_count}/{total_needed} | Closed: {closed_count}/{total_needed}"
        instruction_text = "s: start/stop | o: open class | c: closed class | q: quit"
        
        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        cv2.putText(display, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        cv2.putText(display, instruction_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Eye Data Collection", display)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            collecting = not collecting
            print(f"Collection {'started' if collecting else 'stopped'}")
        
        elif key == ord("o"):
            current_class = "open"
            print("Switched to collecting 'open' eye images")
        
        elif key == ord("c"):
            current_class = "closed"
            print("Switched to collecting 'closed' eye images")
        
        elif key == ord("q"):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("Data collection complete!")
    print(f"Open eye images: {open_count}")
    print(f"Closed eye images: {closed_count}")

if __name__ == "__main__":
    collect_data() 