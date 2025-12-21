import os
import sys
try:
    import speech_recognition as sr
except ImportError:
    print("Please install the 'speechrecognition' package: pip install SpeechRecognition")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("Please install the 'pydub' package: pip install pydub")
    sys.exit(1)

def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "[Could not understand audio]"
    except sr.RequestError as e:
        return f"[Could not request results; {e}]"

def main():
    mp3_path = "Curtis3.mp3"
    if not os.path.isfile(mp3_path) or not mp3_path.lower().endswith('.mp3'):
        print("Invalid file. Please provide a valid mp3 file.")
        return

    wav_path = mp3_path[:-4] + "_temp.wav"
    mp3_to_wav(mp3_path, wav_path)
    print("Transcribing audio...")
    transcript = transcribe_audio(wav_path)
    os.remove(wav_path)

    txt_path = mp3_path[:-4] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcription saved to {txt_path}")

if __name__ == "__main__":
    main()