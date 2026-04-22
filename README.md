# Voice AI Agent - Clinical Appointment Booking

A real-time multilingual voice AI agent for clinical appointment booking.
Built for the 2care.ai assignment.

## Tech Stack
- **STT**: Deepgram Nova-2 (Speech to Text)
- **LLM**: Groq (LLaMA 3 with tool calling)
- **TTS**: ElevenLabs Multilingual v2
- **Backend**: Python + FastAPI
- **Memory**: In-session conversation history

## Setup

### 1 - Clone the repo
```bash
git clone https://github.com/Udaykirankota32/voice-ai-agent
cd voice-ai-agent
```

### 2 - Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3 - Install dependencies
```bash
pip install -r requirements.txt
```

### 4 - Add API keys
Create a `.env` file:
```env
DEEPGRAM_API_KEY=your_key
GROQ_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
```

### 5 - Run
```bash
python app/main.py
```

## Architecture
```text
User Voice -> Deepgram STT -> Groq LLM Agent -> ElevenLabs TTS -> Audio Response
                      |
                      v
                 Tool Calling
            (book / cancel / check)
                      |
                      v
                Session Memory
```

## Latency Breakdown
| Step | Latency |
|------|---------|
| STT (Deepgram) | ~1500-2500ms |
| LLM Agent (Groq) | ~1000-2000ms |
| TTS (ElevenLabs) | ~200-400ms |
| **Total** | **~3000-6000ms** |

## Architectural Decisions

**Why Deepgram?**
Chosen for real-time streaming support, Hindi/Tamil language support, and a simple Python SDK. Nova-2 gives strong accuracy for Indian accents.

**Why Groq?**
Groq provides very low-latency inference for LLaMA-based models, which is important for fast voice workflows.

**Why ElevenLabs?**
High-quality multilingual voice generation with good Hindi and Tamil support.

**Memory Design**
Two-level memory system:
- Session memory: full conversation history passed to the LLM each turn
- Patient memory: stores patient preferences and past appointments

## Tool Calling
The agent uses three tools:
- `book_appointment` - books a slot for a patient
- `check_availability` - checks if doctor is free
- `cancel_appointment` - cancels an existing booking

## Known Limitations
- Latency is above the 450ms target due to network round trips
- Tamil STT accuracy is lower than English/Hindi on Deepgram
- No persistent database - patient memory resets on restart
- No real phone call integration (Twilio not implemented)
- Audio playback requires ffmpeg installation

## What I Would Add With More Time
- Redis for persistent memory with TTL
- WebSocket streaming for lower latency
- Twilio integration for real phone calls
- PostgreSQL for appointments database
- Interrupt/barge-in handling
