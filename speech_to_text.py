import os
import logging
from io import BytesIO

import speech_recognition as sr
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment

# --- Configuration ---
# Load environment variables from a .env file (for GROQ_API_KEY)
load_dotenv()

# Set up clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Main Functions ---

def record_audio(file_path: str, timeout: int = 5, phrase_time_limit: int = None):
    """
    Records audio from the microphone and saves it as an MP3 file.
    The recording stops automatically after a pause in speech.

    Args:
        file_path (str): Path to save the recorded audio file.
        timeout (int): Maximum time to wait for a phrase to start (in seconds).
        phrase_time_limit (int): Maximum time for the phrase to be recorded. If None, it records until a pause.
    """
    recognizer = sr.Recognizer()
    # Increase the pause threshold to 2 seconds to allow for longer pauses
    recognizer.pause_threshold = 1.25
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logging.info("Start speaking now..")
            # Record the audio
            audio_data = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file in memory
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred during recording: {e}")

def transcribe_with_groq(audio_filepath: str) -> str:
    """
    Transcribes an audio file using the Groq API with the Whisper model.

    Args:
        audio_filepath (str): The path to the audio file to transcribe.

    Returns:
        str: The transcribed text.
    """
    try:
        # Check if the API key is available
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY environment variable not set."

        client = Groq(api_key=GROQ_API_KEY) # Pass the key explicitly
        stt_model = "whisper-large-v3"
        
        logging.info(f"Sending '{audio_filepath}' to Groq for transcription...")
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        logging.info("Transcription received.")
        return transcription.text

    except FileNotFoundError:
        return f"Error: The audio file was not found at {audio_filepath}"
    except Exception as e:
        return f"An error occurred during transcription: {e}"

# --- Main Execution Block ---
# This ensures the code only runs when the script is executed directly
if __name__ == "__main__":
    audio_filepath = "recorded_speech.mp3"
    
    # Step 1: Record the audio from the microphone. This was the missing link.
    record_audio(audio_filepath)
    
    # Step 2: Now that the file exists, transcribe it.
    transcribed_text = transcribe_with_groq(audio_filepath)
    
    # Step 3: Print the final result
    print("\n" + "="*30)
    print("Transcription Result:")
    print(transcribed_text)
    print("="*30)

