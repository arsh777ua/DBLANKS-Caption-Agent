import base64
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Disable Gradio telemetry ping which causes Windows socket dropouts
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

# Optional Indic transliteration package
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

TARGET_SAMPLE_RATE: int = 16000
DEFAULT_BATCH_SIZE: int = 16


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Determine hardware device and optimal precision."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


DEVICE, DTYPE = get_device_and_dtype()
print(f"Initializing models on {DEVICE} ({DTYPE})")

# Load VAD model globally once
print("Loading Silero VAD model...")
VAD_MODEL = load_silero_vad()

ASR_MODEL: Optional[Any] = None


def get_asr_model() -> Any:
    """Lazy loader / singleton accessor for Qwen3-ASR model."""
    global ASR_MODEL
    if ASR_MODEL is not None:
        return ASR_MODEL

    if Qwen3ASRModel is None:
        raise RuntimeError("qwen-asr is not installed. Please install requirements.")

    print("Loading Qwen/Qwen3-ASR-0.6B...")
    try:
        ASR_MODEL = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=DTYPE,
            device_map=str(DEVICE),
        )
    except TypeError:
        ASR_MODEL = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            torch_dtype=DTYPE,
            device_map=str(DEVICE),
        )
    except Exception as exc:
        print(f"Fallback load without device_map: {exc}")
        ASR_MODEL = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B")
        if hasattr(ASR_MODEL, "to"):
            ASR_MODEL = ASR_MODEL.to(device=DEVICE, dtype=DTYPE)
    return ASR_MODEL


# Eager warm-up at startup if available
if Qwen3ASRModel is not None:
    try:
        get_asr_model()
    except Exception as err:
        print(f"Warm-up notice: Model will load on first request ({err})")


def check_ffmpeg() -> None:
    """Ensure ffmpeg is in system PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found in system PATH. Please install FFmpeg (e.g. winget install Gyan.FFmpeg) and restart."
        )


def extract_audio_direct(input_file_path: str) -> np.ndarray:
    """
    Extract 16kHz mono float32 PCM directly from audio/video via FFmpeg pipe (RAM-based, no disk write).
    Falls back to temporary file if pipe output fails.
    """
    check_ffmpeg()
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", input_file_path,
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0 and len(proc.stdout) > 0:
        raw_pcm = np.frombuffer(proc.stdout, dtype=np.int16)
        return (raw_pcm.astype(np.float32) / 32768.0)

    # Fallback to temp file method if pipe failed
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
    try:
        cmd_file = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_file_path, "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1", temp_wav_path,
        ]
        res = subprocess.run(cmd_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {res.stderr}")
        data, _ = sf.read(temp_wav_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)
    finally:
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


def segment_audio(
    audio: np.ndarray,
    vad_model: Any,
    max_segment_sec: float = 6.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> List[Dict[str, int]]:
    """Segment audio using Silero VAD, splitting segments exceeding max_segment_sec."""
    audio_tensor = torch.from_numpy(audio).float()
    vad_timestamps = get_speech_timestamps(
        audio_tensor,
        vad_model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=200,
        min_silence_duration_ms=120,
        return_seconds=False,
    )

    max_samples = int(max_segment_sec * sample_rate)
    segments: List[Dict[str, int]] = []

    for seg in vad_timestamps:
        start, end = seg["start"], seg["end"]
        duration = end - start
        if duration <= max_samples:
            segments.append({"start": start, "end": end})
        else:
            splits = math.ceil(duration / max_samples)
            step = math.ceil(duration / splits)
            curr = start
            for _ in range(splits):
                chunk_end = min(curr + step, end)
                if chunk_end > curr:
                    segments.append({"start": curr, "end": chunk_end})
                curr = chunk_end
    return segments


# Transliteration mappings
DEVANAGARI_TO_LATIN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', '्': '', 'ः': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'd', 'ढ़': 'dh', 'फ़': 'f',
    '।': '.', '॥': '.'
}
CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़ड़ढ़फ़"
VOWEL_MATRAS = "ािीुूृेैोौंः"


def romanize_devanagari_word(word: str) -> str:
    """Phonetic converter for Devanagari words to readable Latin Hinglish."""
    res = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == '़':
            ch = ch + '़'
            i += 1
        lat = DEVANAGARI_TO_LATIN.get(ch, ch)
        if any(c in CONSONANTS for c in ch):
            next_ch = chars[i + 1] if i + 1 < len(chars) else None
            if next_ch is None:
                pass
            elif next_ch == '्':
                i += 1
            elif next_ch in VOWEL_MATRAS or next_ch == '़':
                pass
            elif any(c in CONSONANTS for c in next_ch):
                lat += 'a'
        res.append(lat)
        i += 1
    return "".join(res)


def transliterate_to_hinglish(text: str) -> str:
    """Converts Devanagari Hindi words to Romanized Hinglish while leaving English words intact."""
    if not text or not re.search(r'[\u0900-\u097f]', text):
        return text

    if INDIC_TRANSLITERATION_AVAILABLE:
        try:
            words = text.split()
            out = []
            for w in words:
                if re.search(r'[\u0900-\u097f]', w):
                    t = transliterate(w, sanscript.DEVANAGARI, sanscript.ITRANS)
                    t = t.replace("aa", "a").replace("ii", "ee").replace("uu", "oo")
                    t = t.replace("Sh", "sh").replace("shh", "sh")
                    out.append(t.lower())
                else:
                    out.append(w)
            return " ".join(out)
        except Exception:
            pass

    words = text.split()
    return " ".join(romanize_devanagari_word(w) if re.search(r'[\u0900-\u097f]', w) else w for w in words)


def remove_punctuation_marks(text: str) -> str:
    """Removes . , / : ; and Devanagari full stops, cleaning excess spacing."""
    if not text:
        return text
    cleaned = re.sub(r'[\.,/:\;।॥]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()


def split_segment_by_chars(segment: Dict[str, Any], max_chars: int) -> List[Dict[str, Any]]:
    """
    Subdivides caption segments to stay within max_chars (e.g. 7 characters for one-word reel captions).
    Distributes start & end timestamps accurately and proportionally across words.
    """
    text = segment["text"].strip()
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return [segment]

    words = text.split()
    if not words:
        return [segment]

    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0

    for w in words:
        w_len = len(w)
        if curr and (curr_len + 1 + w_len > max_chars):
            chunks.append(" ".join(curr))
            curr = [w]
            curr_len = w_len
        else:
            curr.append(w)
            curr_len = curr_len + 1 + w_len if curr_len > 0 else w_len
    if curr:
        chunks.append(" ".join(curr))

    start_sec = segment["start"]
    end_sec = segment["end"]
    total_dur = max(0.05, end_sec - start_sec)
    total_chars = max(1, sum(len(c) for c in chunks))

    results: List[Dict[str, Any]] = []
    curr_t = start_sec
    for i, c in enumerate(chunks):
        c_dur = total_dur * (len(c) / total_chars)
        sub_end = min(curr_t + c_dur, end_sec) if i < len(chunks) - 1 else end_sec
        results.append({
            "start": round(curr_t, 3),
            "end": round(sub_end, 3),
            "text": c,
        })
        curr_t = sub_end
    return results


def format_srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"


def format_vtt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d}.{ms % 1000:03d}"


def build_subtitle_files(results: List[Dict[str, Any]], base_path: str) -> Tuple[str, str, str, str, str]:
    """Build .srt, .vtt, and .txt files with UTF-8 encoding."""
    srt, vtt, txt = [], ["WEBVTT", ""], []

    for idx, item in enumerate(results, start=1):
        s, e, text = item["start"], item["end"], item["text"].strip()
        if not text:
            continue
        srt.extend([str(idx), f"{format_srt_time(s)} --> {format_srt_time(e)}", text, ""])
        vtt.extend([str(idx), f"{format_vtt_time(s)} --> {format_vtt_time(e)}", text, ""])
        txt.append(text)

    srt_str = "\n".join(srt).strip()
    vtt_str = "\n".join(vtt).strip()
    txt_str = "\n".join(txt).strip()

    p_srt, p_vtt, p_txt = f"{base_path}.srt", f"{base_path}.vtt", f"{base_path}.txt"
    with open(p_srt, "w", encoding="utf-8") as f: f.write(srt_str + "\n")
    with open(p_vtt, "w", encoding="utf-8") as f: f.write(vtt_str + "\n")
    with open(p_txt, "w", encoding="utf-8") as f: f.write(txt_str + "\n")

    return srt_str, txt_str, p_srt, p_vtt, p_txt


def run_transcription(
    media_file: Optional[str],
    language_choice: str,
    script_mode: str,
    strip_punctuation: bool,
    context_text: str,
    max_chars_per_line: int,
    max_segment_sec: float = 3.0,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:
    """End-to-end audio extraction, VAD, batch ASR inference, formatting, and file export."""
    if not media_file:
        raise gr.Error("Please upload an audio or video file.")

    progress(0.05, desc="Loading Qwen3-ASR model...")
    asr_model = get_asr_model()

    # 1. Direct audio extraction
    progress(0.12, desc="Extracting audio (FFmpeg)...")
    audio = extract_audio_direct(media_file)
    dur = len(audio) / TARGET_SAMPLE_RATE
    if dur < 0.1:
        raise gr.Error("Extracted audio is empty or too short.")

    # 2. VAD Segmentation
    progress(0.25, desc="Detecting speech with Silero VAD...")
    raw_segs = segment_audio(audio, VAD_MODEL, max_segment_sec=float(max_segment_sec))
    if not raw_segs:
        step = int(max_segment_sec * TARGET_SAMPLE_RATE)
        raw_segs = [{"start": s, "end": min(s + step, len(audio))} for s in range(0, len(audio), step)]

    # 3. Batch transcription
    lang_arg = None if "Auto" in language_choice else ("Hindi" if "Hindi" in language_choice else "English")
    ctx_arg = context_text.strip() if context_text and context_text.strip() else None

    kwargs: Dict[str, Any] = {}
    if lang_arg: kwargs["language"] = lang_arg
    if ctx_arg: kwargs["context"] = ctx_arg

    total_segs = len(raw_segs)
    raw_results = []
    progress(0.35, desc=f"Transcribing {total_segs} speech segments...")

    for i in range(0, total_segs, DEFAULT_BATCH_SIZE):
        batch = raw_segs[i : i + DEFAULT_BATCH_SIZE]
        slices = [(audio[s["start"] : s["end"]], TARGET_SAMPLE_RATE) for s in batch]

        try:
            outputs = asr_model.transcribe(audio=slices[0], **kwargs) if len(slices) == 1 else asr_model.transcribe(audio=slices, **kwargs)
        except Exception:
            outputs = []
            for sl in slices:
                try: outputs.append(asr_model.transcribe(audio=sl, **kwargs)[0])
                except Exception: outputs.append("")

        for seg, out in zip(batch, outputs):
            txt_val = getattr(out, "text", out["text"] if isinstance(out, dict) and "text" in out else str(out))
            clean = txt_val.strip()

            # Transliterate to Romanized Hinglish if selected
            if "Romanized" in script_mode:
                clean = transliterate_to_hinglish(clean)

            # Strip punctuation if enabled
            if strip_punctuation:
                clean = remove_punctuation_marks(clean)

            raw_results.append({
                "start": seg["start"] / TARGET_SAMPLE_RATE,
                "end": seg["end"] / TARGET_SAMPLE_RATE,
                "text": clean,
            })

        progress(0.35 + 0.55 * (min(i + DEFAULT_BATCH_SIZE, total_segs) / total_segs))

    # 4. Word-by-word / Character limit slicing (e.g. 7 chars for one-word captions)
    final_results = []
    if max_chars_per_line > 0:
        for r in raw_results:
            final_results.extend(split_segment_by_chars(r, int(max_chars_per_line)))
    else:
        final_results = raw_results

    # 5. Build Subtitle Files
    progress(0.95, desc="Generating output files...")
    base_name = os.path.splitext(os.path.basename(media_file))[0]
    out_base = os.path.join(tempfile.gettempdir(), f"captions_{base_name}")
    srt_str, txt_str, p_srt, p_vtt, p_txt = build_subtitle_files(final_results, out_base)

    # Calculate quick dashboard stats
    total_words = sum(len(r["text"].split()) for r in final_results)
    stats_markdown = (
        f"**Duration:** `{round(dur, 1)}s` | "
        f"**Segments:** `{len(final_results)}` | "
        f"**Words:** `{total_words}` | "
        f"**Hardware:** `{DEVICE}`"
    )

    progress(1.0, desc="Done!")
    return stats_markdown, srt_str, txt_str, p_srt, p_vtt, p_txt


# Custom Dashboard CSS with Navbar & footer removal
DASHBOARD_CSS = """
/* Hide 'Built with Gradio' and 'Use via API' footers */
footer, .gradio-container footer, #footer, .footer, a[href*="gradio.app"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

body, .gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Modern Top Navigation Bar */
.navbar-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 24px;
    background: #11111b;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
}

.brand-section {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
    overflow: hidden;
    background: transparent;
}

.brand-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: 8px;
}

.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-badge {
    background: rgba(99, 102, 241, 0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 18px;
}

.nav-link {
    color: #cbd5e1;
    text-decoration: none;
    font-size: 0.92rem;
    font-weight: 500;
    transition: all 0.2s ease;
    padding: 6px 12px;
    border-radius: 8px;
}

.nav-link:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.07);
}

.nav-link.active-link {
    color: #a855f7;
    background: rgba(168, 85, 247, 0.12);
}

.nav-link-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: #ffffff !important;
    text-decoration: none;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
    transition: transform 0.15s ease;
}

.nav-link-btn:hover {
    transform: translateY(-1px);
}

.stat-pill {
    background: #181825;
    border-radius: 8px;
    padding: 8px 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
    font-size: 0.9rem;
    margin-bottom: 12px;
}

.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}
"""


def get_logo_html() -> str:
    """Reads logo.png and encodes it as base64 for seamless inline rendering."""
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f'<img src="data:image/png;base64,{encoded}" alt="Dblanks Logo" />'
        except Exception:
            pass
    # Fallback SVG icon if logo.png is not found
    return """<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 4h7a8 8 0 0 1 8 8 8 8 0 0 1-8 8H4V4z" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M4 12h7" stroke="#ffffff" stroke-width="2.5" stroke-linecap="round"/>
    </svg>"""


def create_ui() -> gr.Blocks:
    """Builds the Dblanks Captions Agent dashboard UI."""
    logo_img_tag = get_logo_html()
    with gr.Blocks(title="Dblanks Captions Agent", theme=gr.themes.Soft()) as demo:
        # Top Navigation Bar
        gr.HTML(
            f"""
            <div class="navbar-container">
                <div class="brand-section">
                    <div class="brand-logo">
                        {logo_img_tag}
                    </div>
                    <h2 class="brand-title">Dblanks Captions Agent</h2>
                    <span class="brand-badge">Qwen3 ASR</span>
                </div>
                <div class="nav-links">
                    <a href="#" class="nav-link active-link">Studio</a>
                    <a href="#settings-anchor" class="nav-link">Settings</a>
                    <a href="https://www.dblanks.site" target="_blank" rel="noopener noreferrer" class="nav-link-btn">
                        Dblanks ↗
                    </a>
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            # Left Column: Upload & Main Controls
            with gr.Column(scale=5):
                gr.Markdown("### 🎙️ 1. Input Media")
                file_input = gr.File(
                    label="Upload Media (.mp3, .wav, .mp4, .mkv)",
                    file_types=[".mp3", ".wav", ".mp4", ".mkv"],
                    type="filepath",
                )

                gr.Markdown("### ⚙️ 2. Quick Caption Presets")
                with gr.Group():
                    script_mode = gr.Radio(
                        label="Hinglish Script Format",
                        choices=[
                            "100% Romanized Hinglish (e.g. 'aaj hum python seekhenge')",
                            "Standard (Devanagari for Hindi + Latin for English)",
                        ],
                        value="100% Romanized Hinglish (e.g. 'aaj hum python seekhenge')",
                        info="Default: Romanized Hinglish for social media reels & shorts.",
                    )

                    strip_punct = gr.Checkbox(
                        label="Remove Punctuation ( . , / : ; )",
                        value=True,
                        info="Default ON: Cleans punctuation marks for seamless video captions.",
                    )

                    char_slider = gr.Slider(
                        minimum=0,
                        maximum=60,
                        value=7,
                        step=1,
                        label="Max Characters Per Line (Default: 7 = One-Word Reel Captions)",
                        info="Set to 7 for punchy one-word dynamic captions, or 25-40 for standard multi-word subtitles. (0 = off)",
                    )

                # Settings section with anchor for the navbar link
                gr.HTML('<div id="settings-anchor"></div>')
                with gr.Accordion("⚙️ Settings & Advanced Controls", open=False):
                    lang_mode = gr.Dropdown(
                        label="Language Detection",
                        choices=["Auto (Best for Hinglish)", "Hindi", "English"],
                        value="Auto (Best for Hinglish)",
                    )
                    context_box = gr.Textbox(
                        label="Hotwords / Brand Terms",
                        placeholder="E.g. Dblanks, Antigravity, Bengaluru, UPI",
                        lines=2,
                    )
                    max_seg_slider = gr.Slider(
                        minimum=1.0,
                        maximum=12.0,
                        value=3.0,
                        step=0.5,
                        label="Max Speech Pause Threshold (seconds)",
                        info="Default 3.0s: Splices fast continuous speech so subtitles stay neat and readable.",
                    )

                generate_btn = gr.Button(
                    "⚡ Generate Subtitles",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

            # Right Column: Dashboard Results & Instant Download
            with gr.Column(scale=6):
                gr.Markdown("### 📊 3. Caption Preview & Files")
                stats_display = gr.Markdown(
                    "**Status:** `Ready` | Upload an audio or video file to begin.",
                    elem_classes=["stat-pill"],
                )

                with gr.Tabs():
                    with gr.TabItem("Generated SRT"):
                        srt_output = gr.Textbox(
                            label="SRT Timestamped Captions",
                            lines=14,
                            placeholder="Generated SRT captions will appear here...",
                            interactive=False,
                        )
                    with gr.TabItem("Plain Text"):
                        txt_output = gr.Textbox(
                            label="Clean Transcript",
                            lines=14,
                            placeholder="Clean plain text will appear here...",
                            interactive=False,
                        )

                gr.Markdown("#### 📥 Instant Download")
                with gr.Row():
                    down_srt = gr.File(label=".SRT Subtitle", interactive=False)
                    down_vtt = gr.File(label=".VTT Web Subtitle", interactive=False)
                    down_txt = gr.File(label=".TXT Transcript", interactive=False)

        # Wire up click event
        generate_btn.click(
            fn=run_transcription,
            inputs=[
                file_input,
                lang_mode,
                script_mode,
                strip_punct,
                context_box,
                char_slider,
                max_seg_slider,
            ],
            outputs=[
                stats_display,
                srt_output,
                txt_output,
                down_srt,
                down_vtt,
                down_txt,
            ],
        )

    return demo


if __name__ == "__main__":
    try:
        app = create_ui()
        app.queue(default_concurrency_limit=2)
        print("\n" + "=" * 60)
        print("Dblanks Captions Agent is LIVE at: http://127.0.0.1:7860")
        print("   Press Ctrl + C in this terminal to stop.")
        print("=" * 60 + "\n")
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            css=DASHBOARD_CSS,
            inbrowser=True,  # Automatically opens your browser
        )
    except Exception as exc:
        print(f"\n[CRASH DETECTED]: {exc}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
