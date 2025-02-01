import cv2
from object_detection import detect_objects

# Dictionary mapping objects to career paths
career_mapping = {
        "mobile": ["Biologist", "Medical Scientist", "Lab Technician"],
    "test_tube": ["Chemist", "Pharmacist", "Chemical Engineer"],
    "laptop": ["Software Engineer", "Data Scientist", "IT Support Specialist"],
    "calculator": ["Accountant", "Financial Analyst", "Actuary"],
    "camera": ["Photographer", "Videographer", "Film Director"],
    "paintbrush": ["Artist", "Graphic Designer", "Art Instructor"],
    "book": ["Librarian", "Author", "Editor"],
    "gavel": ["Judge", "Lawyer", "Paralegal"],
    "stethoscope": ["Doctor", "Nurse", "Healthcare Administrator"],
    "hammer": ["Carpenter", "Construction Manager", "Mechanical Engineer"],
    "spade": ["Landscaper", "Gardener", "Agricultural Engineer"],
    "phone": ["Telecommunications Specialist", "Customer Service Representative", "Sales Executive"],
    "globe": ["Geographer", "Travel Agent", "International Relations Specialist"],
    "mosquito_net": ["Nurse", "Pharmacist", "Healthcare Administrator"]
}

def map_to_career(object_name):
    return career_mapping.get(object_name, ["Unknown Career Path"])

def get_camera_frame():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return None
    
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Release the webcam
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return None
    
    return frame

def start_ar_experience():
    frame = get_camera_frame()
    if frame is None:
        return
    
    object_name, confidence = detect_objects(frame)
    
    if confidence > 0.5:
        print(f"Detected object: {object_name}, Confidence: {confidence}")
        career_paths = map_to_career(object_name)
        print(f"Suggested Careers: {career_paths}")
    else:
        print("No confident detection found")
