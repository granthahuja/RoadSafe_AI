from datetime import datetime


def create_rescue_case(incident):
    """
    Creates a RescueChain response case from a detected incident.
    Emergency response is simulated.
    """

    case = {
        "case_id": f"RS-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "filename": incident.get("filename", "Unknown"),
        "timestamp": incident.get("timestamp"),
        "accident_probability": incident.get(
            "accident_probability", 0.0
        ),
        "incident_priority": incident.get(
            "incident_priority", "Not assessed"
        ),
        "review_flag": incident.get("review_flag", False),
        "status": "Awaiting operator review",
        "operator_verified": False,
        "response_status": "Not initiated"
    }

    return case

def initiate_response(case):
    """
    Initiates a simulated response after operator verification.
    No real emergency services are contacted.
    """

    if not case.get("operator_verified", False):
        return False

    case["response_status"] = "Response initiated"
    case["response_timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    case["status"] = "Response in progress"

    return True

def acknowledge_response(case):
    """
    Records acknowledgement of a simulated response.
    """

    if case.get("response_status") != "Response initiated":
        return False

    case["response_status"] = "Response acknowledged"
    case["acknowledgement_timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    case["status"] = "Response acknowledged"

    return True

def confirm_assistance(case):
    """
    Records simulated hospital confirmation of assistance.
    """

    if case.get("response_status") != "Response acknowledged":
        return False

    case["response_status"] = "Assistance confirmed"
    case["assistance_timestamp"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    case["status"] = "Assistance confirmed"

    return True