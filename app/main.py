import os
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions
from elevenlabs.client import ElevenLabs
from app.agent import run_agent
from app.memory import session_memory

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ─────────────────────────────────────────
# STEP 1 — STT
# Audio file → text
# ─────────────────────────────────────────
def transcribe_audio(file_path: str, language: str = "en") -> str:
    print(f"\n[STT] Transcribing {file_path}...")
    start = time.time()

    deepgram = DeepgramClient(DEEPGRAM_API_KEY)

    with open(file_path, "rb") as audio:
        buffer_data = audio.read()

    payload = {"buffer": buffer_data}

    options = PrerecordedOptions(
        model="nova-2",
        language=language,
        smart_format=True,
    )

    response = deepgram.listen.prerecorded.v("1").transcribe_file(
        payload, options
    )

    transcript = response.results.channels[0].alternatives[0].transcript

    elapsed = (time.time() - start) * 1000
    print(f"[STT] Done in {elapsed:.0f}ms → '{transcript}'")
    return transcript


# ─────────────────────────────────────────
# STEP 2 — TTS
# Text → audio saved as mp3
# ─────────────────────────────────────────
def speak_response(text: str):
    print(f"\n[TTS] Converting to speech: '{text}'")
    start = time.time()

    elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Save to file
    output_path = "response.mp3"
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    elapsed = (time.time() - start) * 1000
    print(f"[TTS] Done in {elapsed:.0f}ms")
    print(f"[TTS] Audio saved to response.mp3 — open it to hear the response!")


# ─────────────────────────────────────────
# FULL PIPELINE
# Audio in → STT → Agent → TTS → Audio out
# ─────────────────────────────────────────
def run_pipeline(audio_file: str, language: str = "en"):
    print("\n" + "="*50)
    print("FULL PIPELINE START")
    print("="*50)

    total_start = time.time()

    # Step 1 — Speech to text
    transcript = transcribe_audio(audio_file, language)

    if not transcript:
        print("[ERROR] Could not understand audio")
        return

    # Step 2 — Update session memory with user message
    session_memory.add_user_message(transcript)

    # Step 3 — Agent thinks using full conversation history
    agent_response = run_agent(transcript, session_memory.get_history())

    # Step 4 — Guard against empty response
    if not agent_response or agent_response == "None":
        agent_response = "I understood your request. Could you please provide more details so I can help you book an appointment?"

    # Step 5 — Save agent response to memory
    session_memory.add_agent_message(agent_response)

    print(f"\n[MEMORY] Session has {len(session_memory.get_history())} messages so far")

    # Step 6 — Text to speech
    speak_response(agent_response)

    total_elapsed = (time.time() - total_start) * 1000
    print(f"\n[LATENCY] Total pipeline: {total_elapsed:.0f}ms")
    print("="*50)


# ─────────────────────────────────────────
# TEST THE FULL PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Clear memory before fresh test
    session_memory.clear()

    # Simulate a multi-turn conversation
    print("Turn 1 — First message")
    run_pipeline("Test-1.m4a", language="en")

    print("\nTurn 2 — Follow up (agent should remember context)")
    run_pipeline("Test-1.m4a", language="en")