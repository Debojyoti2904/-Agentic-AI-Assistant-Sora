import os
import time
import threading
import gradio as gr
import cv2
import atexit
from groq import Groq
from speech_to_text import record_audio, transcribe_with_groq
from ai_agent import ask_agent, llm
from text_to_speech import text_to_speech_with_elevenlabs
from tools import set_last_frame, start_camera, get_current_frame, is_camera_running

# --- CONFIGURATION ---
audio_filepath = "audio_question.mp3"

# --- CAMERA STATE ---
camera = None
is_running = False
is_manual_running = False
is_auto_running = False
last_frame = None

# --- CAMERA FUNCTIONS ---
def initialize_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return False
        for _ in range(10):  # Warm-up frames
            camera.read()
    return True

def start_webcam():
    global is_running, is_manual_running, camera, last_frame
    is_running = True
    is_manual_running = True
    if not initialize_camera():
        is_running = False
        is_manual_running = False
        return None
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        set_last_frame(frame)
        return frame
    return None

def stop_webcam():
    global is_running, is_manual_running, camera, last_frame
    is_manual_running = False
    if camera is not None:
        camera.release()
        camera = None
    is_running = False
    last_frame = None  # clear stored frame
    return gr.Image(value=None) # Explicitly clear the image component

def get_webcam_frame():
    global camera, is_running, last_frame
    if not is_running or camera is None or not camera.isOpened():
        return last_frame
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        set_last_frame(frame)
        return frame
    return last_frame

# --- Automatic capture (returns frame directly instead of .update()) ---
def auto_capture_with_feedback(duration=3):
    global is_auto_running, is_running, is_manual_running, camera
    if is_auto_running or is_manual_running:
        return None
    is_auto_running = True
    is_running = True

    if not initialize_camera():
        is_auto_running = False
        is_running = False
        return None

    ret, frame = camera.read()
    if not ret or frame is None:
        is_auto_running = False
        is_running = False
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    set_last_frame(frame)

    def run():
        global is_auto_running, is_running
        time.sleep(duration)
        if not is_manual_running:
            stop_webcam()
        is_auto_running = False
        if not is_manual_running:
            is_running = False

    threading.Thread(target=run, daemon=True).start()
    return frame

# --- CHAT FUNCTIONS ---
def record_and_respond(chat_history, webcam_output):
    try:
        chat_history.append({"role": "assistant", "content": "Listening... 🎤"})
        yield chat_history, webcam_output

        record_audio(file_path=audio_filepath)
        if not os.path.exists(audio_filepath):
            raise FileNotFoundError("Audio file not found after recording.")

        chat_history[-1] = {"role": "assistant", "content": "Transcribing... 📝"}
        yield chat_history, webcam_output
        user_input = transcribe_with_groq(audio_filepath)

        chat_history.pop() # Remove "Transcribing..."
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": "Thinking... 🧠"})
        yield chat_history, webcam_output

        # --- Check if camera is required ---
        pre_check_prompt = f"""
You are an assistant that decides whether a query absolutely requires camera access.
Answer only 'Yes' or 'No'. If unsure, answer 'No'.
Query: "{user_input}"
"""
        pre_check_response = llm.invoke([{"role": "user", "content": pre_check_prompt}])
        answer = pre_check_response.content.strip().lower()

        new_frame = None
        if "yes" in answer and not is_camera_running():
            new_frame = auto_capture_with_feedback(duration=3)

        response = ask_agent(user_query=user_input)
        text_to_speech_with_elevenlabs(input_text=response, output_filepath="final.mp3")

        chat_history[-1] = {"role": "assistant", "content": response}
        yield chat_history, (new_frame if new_frame is not None else webcam_output)

    except Exception as e:
        error_message = f"An error occurred: {e}"
        if chat_history and chat_history[-1].get("role") == "assistant":
             chat_history[-1]['content'] = error_message
        else:
            chat_history.append({"role": "assistant", "content": error_message})
        yield chat_history, webcam_output

# --- GRADIO UI ---
with gr.Blocks(css="footer {display: none !important}") as demo:
    gr.Markdown("<h1 style='color: orange; text-align: center; font-size: 4em;'> Sora – Your Personal AI Assistant</h1>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="secondary")
            
            webcam_output = gr.Image(label="Live Feed", show_label=False, width=640, height=480, type="numpy", interactive=False)
            webcam_timer = gr.Timer(0.033)
            
        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            # --- FIXED: Removed the avatar_images parameter ---
            chatbot = gr.Chatbot(label="Conversation", height=400, show_label=False, type='messages')
            record_btn = gr.Button("🎤 Record & Chat", variant="primary")
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            
    # --- EVENT HANDLERS ---
    start_btn.click(fn=start_webcam, outputs=webcam_output)
    stop_btn.click(fn=stop_webcam, outputs=webcam_output)
    webcam_timer.tick(fn=get_webcam_frame, outputs=webcam_output, show_progress=False)

    record_btn.click(fn=record_and_respond, inputs=[chatbot, webcam_output], outputs=[chatbot, webcam_output])
    clear_btn.click(fn=lambda: [], outputs=chatbot)

# --- CLEANUP ON EXIT ---
atexit.register(stop_webcam)

# --- LAUNCH ---
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)