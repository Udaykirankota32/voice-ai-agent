import os
import argparse
from deepgram import DeepgramClient, DeepgramClientOptions
from dotenv import load_dotenv

# Load your API keys from .env file
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def transcribe_audio(file_path: str, language: str = "en"):
    """
    Takes an audio file and converts it to text using Deepgram v6.
    language options: "en" (English), "hi" (Hindi), "ta" (Tamil)
    """
    config = DeepgramClientOptions(api_key=DEEPGRAM_API_KEY)
    deepgram = DeepgramClient(config=config)

    with open(file_path, "rb") as audio:
        buffer_data = audio.read()

    payload = {"buffer": buffer_data}

    options = {
        "model": "nova-2",
        "language": language,
        "smart_format": True,
    }

    response = deepgram.listen.prerecorded.v("1").transcribe_file(
        payload, options
    )

    transcript = response.results.channels[0].alternatives[0].transcript
    return transcript


# ---- TEST IT ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file with Deepgram.")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=os.path.join(PROJECT_ROOT, "Test-1.m4a"),
        help="Path to the audio file (default: Test-1.m4a in the project root)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "hi", "ta"],
        help="Language code for transcription (default: en)",
    )
    args = parser.parse_args()

    print("Testing STT pipeline...")
    print("Sending audio to Deepgram...")

    audio_file = args.audio_file
    if not os.path.isabs(audio_file):
        candidate = os.path.join(PROJECT_ROOT, audio_file)
        if os.path.isfile(candidate):
            audio_file = candidate

    if not os.path.isfile(audio_file):
        audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm"}
        available = [
            name
            for name in os.listdir(PROJECT_ROOT)
            if os.path.splitext(name)[1].lower() in audio_extensions
        ]
        available_text = ", ".join(sorted(available)) if available else "None found"
        raise FileNotFoundError(
            f"Audio file not found: {args.audio_file}. Available audio files in project root: {available_text}"
        )

    result = transcribe_audio(audio_file, language=args.language)

    print("\n--- RESULT ---")
    print(f"Transcript: {result}")