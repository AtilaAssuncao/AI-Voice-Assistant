import time
import numpy as np
from soundmanager import SoundManager
from speech_to_text import VAD, STT
from text_to_speech import TTS
from model_request import LLM


class VoiceAssistant:
    
    def __init__(self, stt: STT, llm: LLM, tts: TTS, **options): 
        self._sound_manager = SoundManager()
        self._stt = stt
        self._llm = llm
        self._tts = tts
    
        self._vad = VAD()
        self._speech_buffer = []
        self._last_interaction_time = time.time()

        self._ask_repeat = options.get('ask_repeat', True)

        if options.get('silence_timeout', None) is None or options.get('silence_timeout') is False:
            self._silence_timeout = 2.0
        else:
            self._silence_timeout = options.get('silence_timeout')


    def _process_audio_to_text(self):
        mono_chunk = self._sound_manager.get_buffer_in()[:, 0]

        chunk = (mono_chunk * 32767).astype(np.float16)
        is_speech, _ = self._vad.detect_speech(chunk)

        if is_speech:
            self._speech_buffer.append(chunk)
            self._last_interaction_time = time.time()

            return None
        else:
            if self._speech_buffer and (time.time() - self._last_interaction_time) > (self._silence_timeout):
                
                self._sound_manager.stop_input()

                speech_array = np.concatenate(self._speech_buffer)
                np_audio = speech_array.astype(np.float32) / 32767.0

                transcription = self._stt.transcribe(np_audio)
                self._speech_buffer = []

                return transcription

        return None

    def _get_assistant_query(self, message): 
        if message:
            return self._llm.request(message)
        return None
    
    def _process_text_to_audio(self, message):
        if message:
          audio_array = self._tts.to_speech(message)
          return audio_array
        
        return None

    def _again(self, input: str):
        return True if input == 'y' or input == 'Y' else False

    def _run_with_repetitions(self):
        transcription = None
        is_writing = True
        repeat_in, repeat_out = False, False

        print("🎙️ - listening ... ")
        while not repeat_in:
            transcription = self._process_audio_to_text()

            if transcription:
                print('-', transcription)
                repeat_in = not self._again(input('Speak again? (Y or y) '))
                if not repeat_in:
                    transcription = None
                    self._sound_manager.start_input()

        if transcription:
            response_assistant = self._get_assistant_query(transcription)
            print('-', response_assistant)
            chunk_out = self._process_text_to_audio(response_assistant)

            while chunk_out.any() and not repeat_out:
                print("🔊 - Speaking ...")
                is_writing = self._sound_manager.play_audio(chunk_out)
                repeat_out = not self._again(input('Listen again? (Y or y) '))

            if not is_writing:
                self._sound_manager.start_input()

            print("-"*25)

    def _run_without_repetitions(self):
        print("🎙️ - listening ... ")
        transcription = self._process_audio_to_text()

        if transcription:
            print('- ', transcription)
            response_assistant = self._get_assistant_query(transcription)
            print('- ', response_assistant)
            chunk_out = self._process_text_to_audio(response_assistant)

            if chunk_out.any():
                print("🔊 - Speaking ...")
                is_writing = self._sound_manager.play_audio(chunk_out)

                if not is_writing: # TODO
                    self._sound_manager.start_input()
            
            print("-"*25)

    def run(self):
        print("Running ...")
        self._sound_manager.start_input()
        self._sound_manager.start_output()

        try:
            while True:
                if self._ask_repeat:
                    self._run_with_repetitions()
                else:
                    self._run_without_repetitions()
        except KeyboardInterrupt:
            print("Stopping...")
            self._sound_manager.stop()
            self._sound_manager.close()




if __name__ == '__main__':
    voice_assistant = VoiceAssistant(
        stt=STT(device='cuda'), 
        llm=LLM(model='language-assistant:latest'), 
        tts=TTS(device='cuda')
      )
    voice_assistant.run()
