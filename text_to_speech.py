import os
import elevenlabs
from elevenlabs.client import ElevenLabs
from gtts import gTTS
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables from a .env file
load_dotenv()
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """Generates speech using ElevenLabs and plays the MP3 file with Pydub."""
    print("--- Using ElevenLabs ---")
    print("Generating audio from ElevenLabs...")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio_data = client.text_to_speech.convert(
        text=input_text,
        voice_id="cgSgspJ2msm6clMCkdW9",
        model_id="eleven_multilingual_v2",
        output_format="mp3_22050_32",
    )
    elevenlabs.save(audio_data, output_filepath)
    print(f"Audio saved to {output_filepath}")
    play_audio(output_filepath)

def text_to_speech_with_gtts(input_text, output_filepath):
    """Generates speech using Google Text-to-Speech and plays the MP3 file."""
    print("--- Using gTTS ---")
    print("Generating audio from gTTS...")
    audio_obj = gTTS(text=input_text, lang="en", slow=False)
    audio_obj.save(output_filepath)
    print(f"Audio saved to {output_filepath}")
    play_audio(output_filepath)

def play_audio(filepath):
    """Plays the audio file using Pydub."""
    try:
        print("Loading audio file with Pydub...")
        sound = AudioSegment.from_mp3(filepath)
        print("Playing audio...")
        play(sound)
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred while playing with Pydub: {e}")

# --- Main part of your script ---
# input_text = "Hello guys, I am your friend Sora."
# output_filepath = "test_text_to_speech.mp3"

# --- Choose which function to run ---
# To use ElevenLabs, uncomment the line below
# text_to_speech_with_elevenlabs(input_text, output_filepath)

# To use gTTS, comment out the line above and uncomment the line below
# text_to_speech_with_gtts(input_text, output_filepath)