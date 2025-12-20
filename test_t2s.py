import os


def speak(text):
    os.system(f'flite -t "{text}"')


if __name__ == '__main__':
    speak("testing")
