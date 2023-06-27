from twilio.rest import Client
import cv2
import winsound
from scipy.spatial import distance as dist
from imutils import face_utils

# Constants for eye aspect ratio (EAR) thresholds
EAR_THRESHOLD = 2.0
EAR_CONSEC_FRAMES = 20

# Twilio account credentials
TWILIO_ACCOUNT_SID = 'ACd1deb0b75edb76bbfa51babf29b0c138'
TWILIO_AUTH_TOKEN = 'a18b290043b46f40807dfe83e8ed1888'
TWILIO_PHONE_NUMBER = 'whatsapp:+14'
RECIPIENT_PHONE_NUMBER = 'whatsapp:+91'

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):

    # Convert the eye landmarks to a one-dimensional array
    eye = eye.reshape(-1, 2)

    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize counters and flags
frame_counter = 0
drowsy_flag = False

# Load the Haar cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the video stream
cap = cv2.VideoCapture(0)

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        face_roi = gray[y:y+h, x:x+w]

        # Detect eyes in the face ROI
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Adjust the eye coordinates relative to the face
            eye_x = x + ex
            eye_y = y + ey

            # Extract the eye region of interest (ROI)
            eye_roi = gray[eye_y:eye_y+eh, eye_x:eye_x+ew]

            # Apply Gaussian blur to reduce noise
            eye_roi = cv2.GaussianBlur(eye_roi, (5, 5), 0)

            # Threshold the eye ROI to enhance the features
            _, eye_roi = cv2.threshold(eye_roi, 60, 255, cv2.THRESH_BINARY_INV)

            # Find contours in the eye ROI
            contours, _ = cv2.findContours(eye_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Select the largest contour
                eye_contour = max(contours, key=cv2.contourArea)

                # Compute the eye aspect ratio (EAR)
                eye_ear = eye_aspect_ratio(eye_contour)

                # Draw the eye contour on the frame
                cv2.drawContours(frame, [eye_contour + (eye_x, eye_y)], -1, (0, 255, 0), 1)

                # Check if the EAR is below the threshold
                if eye_ear < EAR_THRESHOLD:
                    frame_counter += 1

                    # If the eyes have been closed for a sufficient number of frames, set the drowsy flag
                    if frame_counter >= EAR_CONSEC_FRAMES:
                        drowsy_flag = True
                        cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Play the Windows alert sound (buzzer)
                        winsound.PlaySound("*", winsound.SND_ALIAS)
                    
                    if drowsy_flag:
                    # Send a WhatsApp message
                        message = client.messages.create(
                        body='Drowsiness detected! Please check the live location below:',
                        from_=TWILIO_PHONE_NUMBER,
                        to=RECIPIENT_PHONE_NUMBER,
                        persistent_action=['geo:22.4690,73.0763|SVIT Vasad'],
                        provide_feedback=True
                    )
                        print('WhatsApp message sent:', message.sid)
                        drowsy_flag = False


                else:
                    frame_counter = 0
                    drowsy_flag = False

                # Draw the computed eye aspect ratio (EAR) on the frame
                cv2.putText(frame, "EAR: {:.2f}".format(eye_ear), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
