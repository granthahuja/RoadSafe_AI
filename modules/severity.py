from ultralytics import YOLO

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

PERSON_CLASS = 0

# Load the model once and reuse it
_visual_model = None


def load_visual_model():
    global _visual_model

    if _visual_model is None:
        _visual_model = YOLO("yolov8n.pt")

    return _visual_model


def analyze_frame(model, frame):
    results = model.predict(
        source=frame,
        conf=0.35,
        verbose=False
    )

    vehicles = []
    people = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id in VEHICLE_CLASSES:
                vehicles.append({
                    "type": VEHICLE_CLASSES[class_id],
                    "confidence": round(confidence, 3)
                })

            elif class_id == PERSON_CLASS:
                people += 1

    return {
        "vehicle_count": len(vehicles),
        "person_count": people,
        "vehicles": vehicles
    }


def preliminary_assessment(result):
    """
    Object-count-based visual activity indicator.
    Not a validated accident severity classifier.
    """

    vehicles = result["vehicle_count"]
    people = result["person_count"]

    if vehicles >= 3 or people >= 3:
        activity = "High visual activity"
    elif vehicles >= 2 or people >= 2:
        activity = "Moderate visual activity"
    elif vehicles == 1 or people == 1:
        activity = "Low visual activity"
    else:
        activity = "Insufficient visual evidence"

    return activity

def assess_incident(accident_probability, visual_activity):
    """
    Generates preliminary incident priority and an operator review flag.
    Not a validated accident or medical severity classifier.
    """

    if accident_probability >= 0.75:
        if visual_activity == "High visual activity":
            priority = "High"
        else:
            priority = "Moderate"

    elif accident_probability >= 0.50:
        priority = "Moderate"

    else:
        priority = "Low"

    # Independent visual review flag
    review_flag = visual_activity == "High visual activity"

    return {
        "priority": priority,
        "review_flag": review_flag,
        "review_reason": (
            "High visual activity detected by YOLO"
            if review_flag
            else "No high visual activity flag"
        ),
        "assessment_basis": (
            "MobileNetV2 accident probability and YOLO visual activity"
        ),
        "requires_operator_review": True
    }