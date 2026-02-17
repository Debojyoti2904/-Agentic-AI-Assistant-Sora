import cv2
import base64
import threading
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Camera state for manual mode ---
camera = None
last_frame = None
lock = threading.Lock()

def start_camera(camera_id=0):
    global camera
    if camera is None:
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            return False
        for _ in range(10):  # Warm-up frames
            camera.read()
    return True

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def is_camera_running():
    global camera
    return camera is not None and camera.isOpened()

def set_last_frame(frame):
    global last_frame
    with lock:
        last_frame = frame

def get_current_frame():
    global camera, last_frame
    with lock:
        if camera is None or not camera.isOpened():
            return None
        ret, frame = camera.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame
            return frame
    return last_frame

# --- Encode any frame (numpy image) into base64 JPEG ---
def encode_frame_to_base64(frame: np.ndarray) -> str:
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        raise RuntimeError("Failed to encode frame.")
    return base64.b64encode(buf).decode("utf-8")

# --- Capture from camera if running ---
def capture_image_from_camera() -> str:
    global last_frame
    with lock:
        if last_frame is None:
            raise RuntimeError("Camera not started. Please start the camera first.")
        return encode_frame_to_base64(last_frame)

# --- Handle either uploaded/pasted image OR live camera ---
def prepare_image_for_analysis(uploaded_image: np.ndarray = None) -> str:
    """
    If uploaded_image is provided, use it.
    Otherwise, fallback to last camera frame.
    Returns a base64-encoded JPEG string.
    """
    if uploaded_image is not None:
        if not isinstance(uploaded_image, np.ndarray):
            raise ValueError("Uploaded image must be a numpy array.")
        return encode_frame_to_base64(uploaded_image)
    return capture_image_from_camera()

# --- Image analysis tool for LangChain ---
def get_analyze_image_with_query():
    from langchain.tools import Tool
    from groq import Groq

    def analyze_image_with_query(query: str, uploaded_image: np.ndarray = None) -> str:
        img_b64 = prepare_image_for_analysis(uploaded_image)
        model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        client = Groq()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]
        chat_completion = client.chat.completions.create(messages=messages, model=model)
        return chat_completion.choices[0].message.content

    return Tool(
        name="analyze_image_with_query",
        func=analyze_image_with_query,
        description="Use this tool to answer queries that require either a webcam capture or an uploaded image."
    )
