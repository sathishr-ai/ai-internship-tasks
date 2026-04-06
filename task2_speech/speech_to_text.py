# speech_recognition.py
# Requirements: pip install openai-whisper ffmpeg

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import speech_recognition as sr

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using the SpeechRecognition library.
    Args:
        audio_file_path (str): Path to the audio file (WAV format recommended)
    Returns:
        str: Transcribed text.
    """
    # Initialize the recognizer
    r = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file_path) as source:
        audio_data = r.record(source)
    
    # Transcribe using Google Web Speech API (Free, no key required)
    text = r.recognize_google(audio_data)
    return text

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.terminal_style import style

    # Task title
    style.print_header("Speech to Text Transcription")

    audio_file = "sample_audio.wav"
    
    try:
        style.print_input_panel(f"Audio File: {audio_file}", "INPUT")
        transcription = transcribe_audio(audio_file)
        style.print_output_panel(transcription, "OUTPUT")
    except Exception:
        pass