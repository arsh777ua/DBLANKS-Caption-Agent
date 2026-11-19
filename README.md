# ⚡ Dblanks Captions Agent

A local AI-powered studio web application that generates timestamped captions (`.srt`, `.vtt`) and clean transcripts (`.txt`) from uploaded `.mp3`, `.wav`, `.mp4`, and `.mkv` files using **Qwen/Qwen3-ASR-0.6B** with **Hinglish** code-switching and **Silero VAD**.

Built for short-form content creators (Reels, Shorts, TikTok) and long-form video subtitle workflows.

---

## ✨ Features

- **Qwen3-ASR Multilingual Engine**: State-of-the-art Hindi, English, and zero-shot Hinglish speech recognition.
- **100% Romanized Hinglish Mode**: Automatically transliterates Devanagari Hindi to Latin script (e.g. *"aaj hum python seekhenge"* instead of *"आज हम Python सीखेंगे"*).
- **One-Word / Dynamic Reel Captions**: Word-level character slider (default: `7` chars) with proportional timestamp allocation.
- **Punctuation Stripper**: Clean out `. , / : ;` and Devanagari full-stop dandas for modern punchy social captions.
- **Zero-Disk In-Memory Extraction**: Direct FFmpeg stdout pipe into RAM-based NumPy float32 PCM arrays.
- **Studio Dashboard UI**: Clean dark theme with live stats, instant previews, tabbed outputs, and one-click downloads.

---

## 🛠️ Prerequisites

### Install FFmpeg
FFmpeg is required to extract 16kHz mono audio from media files. Ensure it is accessible in your system's `PATH`.

- **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *(Or `choco install ffmpeg`)*
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/dblanks-captions-agent.git
cd dblanks-captions-agent
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch
- **CUDA 12.1+ (NVIDIA GPU)**:
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **CPU / Apple Silicon (MPS)**:
  ```bash
  pip install torch torchaudio
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
python app.py
```

Open your browser to:
👉 **`http://127.0.0.1:7860`**

---

## ⚙️ Performance & Configuration

| Preset | Default Value | Purpose |
|---|---|---|
| **Script Format** | `100% Romanized Hinglish` | Transliterates Hindi words to Latin Hinglish |
| **Max Characters / Line** | `7` | Single / two-word reel captions |
| **Punctuation Stripping** | `Enabled` | Strips `. , / : ;` for clean aesthetic |
| **Speech Pause Threshold** | `3.0s` | Splits unbroken speech chunks into neat lines |

---

## 📄 License
MIT License. Free for personal and commercial use.
