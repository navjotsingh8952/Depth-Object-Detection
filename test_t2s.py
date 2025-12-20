import pyttsx3

_engine = pyttsx3.init()
_engine.setProperty("rate", 150)
_engine.setProperty("volume", 1.0)

def speak(text: str):
    """
    Speak text using offline TTS
    """
    _engine.say(text)
    _engine.runAndWait()
