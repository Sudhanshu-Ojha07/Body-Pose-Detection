#install all these libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe drawing and pose utilities, mediapipe provide pre-trained model for hand tracking, pose detection in this case.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# this is a Function to calculate the angle between three points.
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

# Curl counter variables
counter = 0 
stage = None

# Open the video capture stream
cap = cv2.VideoCapture(0)
# Set camera resolution (increase width and height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720 pixels


# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up mediapipe pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame= cv2.flip(frame,1)

        # Check if the frame was properly captured
        if not ret:
            print("Error: Frame not captured.")
            break

        # Recolor the image from BGR to RGB for mediapipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Improve performance by disabling writeability
        image.flags.writeable = False 

        # Make pose detection
        results = pose.process(image)

        # Recolor the image back to BGR for OpenCV rendering.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates of shoulder, elbow, and wrist for angle calculation.
            try:
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate the angle between shoulder, elbow, and wrist
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize the angle on the video feed
                cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter algorithm
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

            except Exception as e:
                print("Error in processing landmarks:", e)

        # Render the pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the curl counter on the image
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the current stage
        cv2.putText(image, 'STAGE', (65, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the video feed with Mediapipe landmarks
        cv2.imshow('Mediapipe Pose Feed', image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
