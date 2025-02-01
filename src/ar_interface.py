import cv2
from object_detection import detect_objects
from career_mapping import map_to_career
from object_detection import detect_objects, load_my_model
def display_career_information(frame, object_name, confidence):
    careers = map_to_career(object_name)
    career_info = ', '.join(careers)

    # Display the detected object and confidence score on the camera feed
    cv2.putText(frame, f"Object: {object_name} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the associated career paths
    cv2.putText(frame, f"Careers: {career_info}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def start_ar_experience():
    load_my_model()  # Correctly call the function without arguments
    print("Starting AR Career Explorer...")
    
    # Your existing code to open the camera and process frames
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        object_name, confidence = detect_objects(frame)

        # Display the career information based on detected objects
        display_career_information(frame, object_name, confidence)
        
        # Show the camera feed with the object and career information overlay
        cv2.imshow('AR Career Explorer', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()