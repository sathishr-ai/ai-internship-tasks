# speech_recognition.py
# Requirements: pip install openai-whisper ffmpeg

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
    # Provide the path to your audio clip
    audio_file = "sample_audio.wav"
    try:
        print(f"Transcribing {audio_file}...")
        transcription = transcribe_audio(audio_file)
        print("Transcription:")
        print(transcription)
    except sr.UnknownValueError:
        print("Could not understand audio (No speech detected).")
        print("Try recording yourself and saving it as 'sample_audio.wav'!")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the audio file is in WAV format.")