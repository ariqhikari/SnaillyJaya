from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import av
import librosa
from pathlib import Path
import uuid

# --- Inisialisasi Model (Hanya dilakukan SEKALI) ---
print("🚀 Menginisialisasi komponen Video Captioning...")
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Model Captioning ==================
GIT_MODEL_PATH = "public/models/VIDgit_captioning"
PROCESSOR, MODEL = None, None

try:
    if os.path.exists(GIT_MODEL_PATH):
        print("🔄 Memuat GIT model dari lokal...")
        PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_PATH)
        MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_PATH).to(DEVICE)
    else:
        print("⬇️ Mengunduh GIT model dari Hugging Face...")
        PROCESSOR = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
        MODEL = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex").to(DEVICE)

        print("💾 Menyimpan model GIT...")
        PROCESSOR.save_pretrained(GIT_MODEL_PATH)
        MODEL.save_pretrained(GIT_MODEL_PATH)
except Exception as e:
    print(f"❌ Gagal memuat GIT model: {e}")
    PROCESSOR, MODEL = None, None

# ================== Model Speech to Text (Whisper) ==================
WHISPER_MODEL_PATH = "public/models/STT_captioning"
ASR = None
try:
    if os.path.exists(WHISPER_MODEL_PATH):
        print("🔄 Memuat Whisper dari lokal...")
        ASR = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_PATH, device=0 if DEVICE.type=="cuda" else -1)
    else:
        print("⬇️ Mengunduh Whisper model...")
        ASR = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=0 if DEVICE.type=="cuda" else -1)

        print("💾 Menyimpan Whisper...")
        os.makedirs(WHISPER_MODEL_PATH, exist_ok=True)
        ASR.model.save_pretrained(WHISPER_MODEL_PATH)
        ASR.tokenizer.save_pretrained(WHISPER_MODEL_PATH)
except Exception as e:
    print(f"❌ Gagal memuat Whisper model: {e}")
    ASR = None

# ================== Fungsi Audio ==================
def extract_audio(video_path: str, audio_out: str) -> str:
    """Ekstrak audio dari video ke file WAV"""
    try:
        container = av.open(video_path)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            print(f"[WARNING] Tidak ada audio di {video_path}")
            return ""
        audio_stream = audio_streams[0]

        out = av.open(audio_out, 'w')
        out.add_stream(template=audio_stream)
        for packet in container.demux(audio_stream):
            out.mux(packet)
        out.close()
        container.close()
        return audio_out
    except Exception as e:
        print(f"[ERROR] Gagal ekstrak audio dari {video_path}: {e}")
        return ""

def transcribe_audio(audio_path: str) -> str:
    """Transkripsi audio dengan Whisper + progress bar & segmen log"""
    if not ASR:
        return "[ERROR] Model ASR tidak dimuat"
    try:
        # Pastikan format WAV
        if not audio_path.lower().endswith(".wav"):
            wav_path = os.path.splitext(audio_path)[0] + ".wav"
            print(f"🔄 Konversi {audio_path} ke {wav_path} ...")
            new_wav = extract_audio(audio_path, wav_path)
            if new_wav:
                audio_path = new_wav
            else:
                return "[ERROR] Konversi ke WAV gagal"

        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)

        # Jalankan Whisper dengan timestamps
        result = ASR({"array": speech, "sampling_rate": sr}, return_timestamps=True)

        # Kalau hasil ada "chunks"
        if isinstance(result, dict) and "chunks" in result:
            chunks = result["chunks"]
            final_text = []

            print("📝 Proses transkripsi per segmen:")
            with tqdm(total=len(chunks), desc="Transcribing", unit="segmen") as pbar:
                for i, seg in enumerate(chunks):
                    start, end = seg.get("timestamp", (None, None))
                    text = seg.get("text", "").strip()
                    print(f"  🔹 [{i+1}] {start:.2f}s - {end:.2f}s: {text}")
                    final_text.append(text)
                    pbar.update(1)

            return " ".join(final_text)

        # Kalau pipeline tidak mendukung chunks
        return result.get("text", "")

    except Exception as e:
        print(f"[ERROR] Gagal transkrip {audio_path}: {e}")
        return ""
# def transcribe_audio(audio_path: str) -> str:
#     """Transkripsi audio dengan Whisper"""
#     if not ASR:
#         return "[ERROR] Model ASR tidak dimuat"
#     try:
#         # Pastikan format WAV
#         if not audio_path.lower().endswith(".wav"):
#             wav_path = os.path.splitext(audio_path)[0] + ".wav"
#             print(f"🔄 Konversi {audio_path} ke {wav_path} ...")
#             new_wav = extract_audio(audio_path, wav_path)
#             if new_wav:
#                 audio_path = new_wav
#             else:
#                 return "[ERROR] Konversi ke WAV gagal"

#         # Load & transkrip dengan Whisper
#         speech, sr = librosa.load(audio_path, sr=16000)
#         result = ASR({"array": speech, "sampling_rate": sr})
#         return result["text"]

#     except Exception as e:
#         print(f"[ERROR] Gagal transkrip {audio_path}: {e}")
#         return ""

# ================== Fungsi Video ==================
def _extract_frames(video_path, num_frames:int=16):
    try:
        container = av.open(video_path)
        frames = []
        total_frames = container.streams.video[0].frames

        if total_frames == 0:
            return []
        step = max(total_frames // num_frames, 1)

        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                img = frame.to_image().convert("RGB")
                frames.append(img)
            if len(frames) >= num_frames:
                break
        container.close()
        return frames
    except Exception as e:
        print(f"[SKIP] Gagal ekstrak frame {os.path.basename(video_path)}: {e}")
        return []

def _generate_single_caption(video_path: str):
    """Buat caption + transcript dari 1 video"""
    if not PROCESSOR or not MODEL:
        return "[ERROR] Model GIT tidak dimuat", "[ERROR] Model ASR tidak dimuat"

    # Caption dengan GIT
    try:
        frames = _extract_frames(video_path, num_frames=16)
        if not frames:
            caption = "[ERROR] Tidak ada frame valid"
        else:
            inputs = PROCESSOR(images=frames, return_tensors="pt").to(DEVICE)
            generated_ids = MODEL.generate(
                pixel_values=inputs.pixel_values,
                temperature=1.0,
                length_penalty=1.0,
                repetition_penalty=1.5,
                min_length=1,
                num_beams=5,
                max_length=50,
            )
            caption = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        caption = f"[ERROR] Caption gagal: {e}"

    # Transcript dengan Whisper
    try:
        temp_audio = f"temp_{uuid.uuid4().hex}.wav"
        audio_path = extract_audio(video_path, temp_audio)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            os.remove(audio_path) if os.path.exists(audio_path) else None
        else:
            transcript = "[WARNING] Tidak ada audio"
    except Exception as e:
        transcript = f"[ERROR] Transcript gagal: {e}"

    return caption, transcript

# ================== Fungsi Utama ==================
def generate_captions_parallel(video_paths, max_workers=4):
    """Proses banyak video secara paralel"""
    results = {}
    print(f"📂 Ditemukan {len(video_paths)} video. Mulai captioning...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_generate_single_caption, path): path for path in video_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                caption, transcript = future.result()
                results[path] = {"Caption": caption, "Transcript": transcript}
                print(f"[SUCCESS] {os.path.basename(path)}")
            except Exception as e:
                results[path] = {"Caption": f"[ERROR] {e}", "Transcript": f"[ERROR] {e}"}
                print(f"[ERROR] {os.path.basename(path)}: {e}")
    return results

def caption_videos_in_folder(video_folder_path: str, delete_corrupted: bool=False) -> pd.DataFrame:
    """Caption + Transcript semua video di folder"""
    if not MODEL:
        print("❌ Model GIT tidak dimuat.")
        return pd.DataFrame()

    folder_path = Path(video_folder_path)
    if not folder_path.is_dir():
        print(f"❌ Folder '{video_folder_path}' tidak ditemukan")
        return pd.DataFrame()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"⚠️ Tidak ada video ditemukan di '{video_folder_path}'")
        return pd.DataFrame()

    results = generate_captions_parallel(video_files, max_workers=4)

    df = pd.DataFrame([
        {"Filename": os.path.basename(v), "Caption": res["Caption"], "Transcript": res["Transcript"]}
        for v, res in results.items()
    ])
    return df
