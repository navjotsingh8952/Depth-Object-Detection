from gtts import gTTS
import os
import uuid

def speak(text):
    file = f"/tmp/{uuid.uuid4().hex}.mp3"
    gTTS(text=text, lang="en").save(file)
    os.system(f"mpg123 {file}")

if __name__ == '__main__':
    speak("testing")