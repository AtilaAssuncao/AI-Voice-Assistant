import torch
import whisper


class VAD:
    
    def __init__(self, threshold=0.5, sampling_rate=16000):
        self._model, self._utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
        self._get_speech_timestamps, _, _, _, _ = self._utils
        self._threshold = threshold
        self._sampling_rate = sampling_rate


    def detect_speech(self, audio_chunk):
        speech_timestamps = self._get_speech_timestamps(audio_chunk, self._model, 
                                                       threshold=self._threshold, sampling_rate=self._sampling_rate)
        return len(speech_timestamps) > 0, speech_timestamps




class STT:

    def __init__(self, language="en", name_model='base', device=None, temperature=None):
        self._model = whisper.load_model(name_model, device)
        self._language = language
        self._temperature = temperature


    def transcribe(self, audio_data):
        if audio_data.any():
            temperature = (0.1, 0, 0.1, 0) if self._temperature is None else self._temperature
            transcription = self._model.transcribe(audio_data, language=self._language, temperature=temperature)['text']
            return transcription
        return None

