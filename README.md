# Offline Command Line AI Adventure Story Game

## Setup Instructions

### 1. Ollama setup

[Install Ollama](https://ollama.com/download)

Install the required models

```bash
ollama pull deepseek-r1:14b
ollama pull llava:latest
```

### 2. Repo setup

Clone this repo.

Create the env

```bash
python -m venv env
pip install -r requirements.txt
```

### 3. Whisper setup

[Install Whisper to the whisper folder](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)

### 4. Run the game

```bash
python .\game.py
```

### Notes

These install commands are for Windows. Other OS's may be different.

The performance of the game depends on your machine's specs. Adjust the AI model specs if needed.
