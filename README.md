## AI Voice Assistant

AI voice assistant that runs on CLI. This project uses speech-to-text recognition and transcription models and text-to-speech models to communicate with the text generation model, and they are used together, thus transforming speech into an input for the text model and transforming its output into audio.


## Getting Started

### Prerequisites
This project uses [Ollama](https://ollama.com/) as a LLM service, such as the DeepSeek deepseek-r1:1.5b model. So with Ollama installed, install the model.

In the terminal
```
ollama run deepseek-r1:1.5b
```


### Installing

Clone the project:
```
  git clone https://github.com/AtilaAssuncao/AI-Voice-Assistant.git
  cd AI-Voice-Assistant
```
Create a virtual environment and activate the environment:
```
  python -m venv .venv &
  source .venv\Scripts\activate
```

Install dependencies:
```
  pip install silero-vad openai-whisper sounddevice ollama
```
And 
```
  pip install git+https://github.com/myshell-ai/MeloTTS.git
  python -m unidic download
```

If your machine has a video card and it supports CUDA, access the [PyTorch](https://pytorch.org/) website to install it with CUDA support.

<hr>

\*If you have any problems installing the dependencies, access the documentation for the [Whisper](https://github.com/openai/whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [MeloTTS](https://github.com/myshell-ai/MeloTTS/tree/main), [Ollama](https://github.com/ollama/ollama) dependencies.

## Running 

```
  python src/main.py
```


## Models used
[Whisper](https://github.com/openai/whisper), [Silero VAD](https://github.com/snakers4/silero-vad), [MeloTTS](https://github.com/myshell-ai/MeloTTS/tree/main).
