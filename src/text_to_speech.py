from melo.api import TTS as _MeloTTS 
from melo.utils import get_text_for_tts_infer

import re
import torch

# TODO: Review
import warnings
warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning
)
from transformers import logging
logging.set_verbosity_error()


SDP_RATIO= 0.2
NOISE_SCALE= 0.6
NOISE_SCALE_W= 0.8
SPEED= 1.0


class MeloTTS(_MeloTTS): 
    
    def __init__(self, language, device='auto', use_hf=True, config_path=None, ckpt_path=None):
        super().__init__(language, device, use_hf, config_path, ckpt_path)


    def tts(self, text, speaker_id=0, speed=1.0):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, True)

        audio_list = []
        for txt in texts:
            if language in ['EN', 'ZH_MIX_EN']:
                txt = re.sub(r'([a-z])([A-Z])', r'\1 \2', txt)
            
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(txt, language, 
                                                                            self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                    x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, ja_bert, 
                    sdp_ratio=SDP_RATIO, noise_scale=NOISE_SCALE, noise_scale_w=NOISE_SCALE_W, length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers

            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        return audio




class TTS:
    
    def __init__(self, language='EN', device="auto", speaker_index=None, speed=0.95):
        self._model = MeloTTS(language=language, device=device)
        self._speed = speed
        self._speaker_index = speaker_index


    def speaker_list(self):
        print(self._model.hps.data.spk2id)

    @property
    def speaker(self):
        return self._model.hps.data.spk2id
    
    @speaker.setter
    def speaker(self, speaker):
        self._speaker_index = speaker
    
    @property
    def speed(self):
        return self._speed
      
    @speed.setter
    def speed(self, speed):
        self._speed = speed
    
    def to_speech(self, speech_text):
        if speech_text:
          speaker_index = 'EN-US' if self._speaker_index is None else self._speaker_index
          return self._model.tts(speech_text, 
                                speaker_id=self._model.hps.data.spk2id[speaker_index], speed=self._speed)
      
        return None
    











