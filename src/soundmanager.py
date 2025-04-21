import queue
import sounddevice as sd
from numpy import ndarray


class SoundManager:

    def __init__(self, channels=1, samplerate_in=16000, blocksize_in=8000, samplerate_out=44100):  
        self.audio_in = queue.Queue()

        def callback_in(indata: ndarray, frames, time, status):
            self.audio_in.put(indata.copy())
                
        self.stream_in = sd.InputStream(callback=callback_in, channels=channels, samplerate=samplerate_in, blocksize=blocksize_in,  dtype='float32') 
        self.stream_out = sd.OutputStream(samplerate=samplerate_out, channels=channels)
        

    def start_input(self):
        self.stream_in.start()

    def stop_input(self, ignore_errors=True):
        self.stream_in.stop(ignore_errors)

    def start_output(self):
        self.stream_out.start()


    def start(self):
        self.stream_in.start()
        self.stream_out.start()

    def stop(self, ignore_errors=True):
        self.stream_in.stop(ignore_errors)
        self.stream_out.stop(ignore_errors)

    def close(self, ignore_errors=True):
        self.stream_in.close(ignore_errors)
        self.stream_out.close(ignore_errors)

    
    def get_buffer_in(self):
        return self.audio_in.get()

    def play_audio(self, chunk):
        return self.stream_out.write(chunk)

