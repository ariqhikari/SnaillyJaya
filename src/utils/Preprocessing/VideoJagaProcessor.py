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
        PROCESSOR.save_pretrained(GIT_MODEL_PATH)
        MODEL.save_pretrained(GIT_MODEL_PATH)
except Exception as e:
    print(f"❌ Gagal memuat GIT model: {e}")

# ================== Model Speech to Text (Whisper) ==================
WHISPER_MODEL_PATH = "public/models/STT_captioning"
ASR = None
try:
    if os.path.exists(WHISPER_MODEL_PATH):
        print("🔄 Memuat Whisper dari lokal...")
        ASR = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_PATH,
                       device=0 if DEVICE.type == "cuda" else -1)
    else:
        print("⬇️ Mengunduh Whisper model...")
        ASR = pipeline("automatic-speech-recognition", model="openai/whisper-medium",
                       device=0 if DEVICE.type == "cuda" else -1)
        os.makedirs(WHISPER_MODEL_PATH, exist_ok=True)
        ASR.model.save_pretrained(WHISPER_MODEL_PATH)
        ASR.tokenizer.save_pretrained(WHISPER_MODEL_PATH)
except Exception as e:
    print(f"❌ Gagal memuat Whisper model: {e}")

# ================== Fungsi Audio ==================
def extract_audio(video_path: str, audio_out: str) -> str:
    try:
        container = av.open(video_path)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
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
        print(f"[ERROR] Ekstrak audio gagal: {e}")
        return ""

def transcribe_with_segments(audio_path: str):
    """Transkripsi audio + ambil timestamp segmen (fallback kalau tidak ada chunks)"""
    if not ASR:
        return []

    # Konversi ke WAV (16kHz mono) kalau belum
    if not audio_path.lower().endswith(".wav"):
        wav_path = os.path.splitext(audio_path)[0] + ".wav"
        new_wav = extract_audio(audio_path, wav_path)
        audio_path = new_wav if new_wav else audio_path

    # Load audio
    speech, sr = librosa.load(audio_path, sr=16000)

    # Jalankan ASR
    result = ASR({"array": speech, "sampling_rate": sr})

    chunks = []
    # Kalau hasil punya segmen (chunks)
    if isinstance(result, dict) and "chunks" in result:
        for seg in result["chunks"]:
            start, end = seg.get("timestamp", (None, None))
            text = seg.get("text", "").strip()
            if start is not None and end is not None and text:
                chunks.append({
                    "start": float(start),
                    "end": float(end),
                    "text": text
                })
    else:
        # Fallback → kalau tidak ada segmen, tetap simpan full text
        chunks.append({
            "start": 0.0,
            "end": len(speech) / sr,
            "text": result["text"].strip() if isinstance(result, dict) else str(result)
        })

    return chunks

# def transcribe_with_segments(audio_path: str):
#     """Transkripsi + ambil timestamp segmen"""
#     if not ASR:
#         return []
#     if not audio_path.lower().endswith(".wav"):
#         wav_path = os.path.splitext(audio_path)[0] + ".wav"
#         new_wav = extract_audio(audio_path, wav_path)
#         audio_path = new_wav if new_wav else audio_path

#     speech, sr = librosa.load(audio_path, sr=16000)
#     result = ASR({"array": speech, "sampling_rate": sr}, return_timestamps=True)

#     chunks = []
#     if isinstance(result, dict) and "chunks" in result:
#         for seg in result["chunks"]:
#             start, end = seg.get("timestamp", (None, None))
#             text = seg.get("text", "").strip()
#             if start is not None and end is not None and text:
#                 chunks.append({
#                     "start": float(start),
#                     "end": float(end),
#                     "text": text
#                 })
#     return chunks

# ================== Fungsi Video ==================
def extract_frame_at(video_path, timestamp):
    """Ambil 1 frame dari video pada detik tertentu"""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        container.seek(int(timestamp * stream.time_base.denominator))
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            container.close()
            return img
        container.close()
    except Exception as e:
        print(f"[ERROR] Gagal ambil frame di {timestamp}s: {e}")
    return None

def caption_frame(image):
    try:
        inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
        generated_ids = MODEL.generate(
            pixel_values=inputs.pixel_values,
            temperature=1.0,
            repetition_penalty=1.5,
            num_beams=3,
            max_length=50,
        )
        return PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return f"[ERROR] Caption gagal: {e}"

def _generate_single_caption(video_path: str):
    # --- KODE LAMA ---
    # Validasi model
    # if not PROCESSOR or not MODEL:
    #     # return []
    #     print("[ERROR] Model captioning belum dimuat.")
    #     return [{"timestamp": 0, "visual": "[ERROR] Model captioning belum dimuat."}]
    # Validasi file video
    # if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
    #     print(f"[ERROR] File video tidak ditemukan atau kosong: {video_path}")
    #     return [{"timestamp": 0, "visual": "[ERROR] File video tidak ditemukan atau kosong."}]
    # results = []
    # try:
    #     # Buka video dan dapatkan durasi
    #     container = av.open(video_path)
    #     duration = float(container.duration / av.time_base)  # durasi detik
    #     container.close()
    #     # Tentukan timestamp (awal, tengah, akhir)
    #     timestamps = [0, duration / 2, max(0, duration - 1)]
    #     for ts in timestamps:
    #         frame = extract_frame_at(video_path, ts)
    #         if frame:
    #             caption = caption_frame(frame)
    #             if not caption or caption.strip() == "":
    #                 print(f"[WARNING] Caption kosong di timestamp {ts}, isi dengan placeholder.")
    #                 caption = "[NO CAPTION]"
    #         else:
    #             caption = "[WARNING] Frame tidak ditemukan"
    #         results.append({
    #             "timestamp": ts,
    #             "visual": caption
    #         })
    # except Exception as e:
    #     print(f"[ERROR] Gagal generate caption untuk {video_path}: {e}")
    #     # results = []
    #     results.append({"timestamp": 0, "visual": f"[ERROR] {str(e)}"})
    # return results

    # --- KODE BARU ---
    results = []
    if not PROCESSOR or not MODEL:
        print("[ERROR] Model captioning belum dimuat.")
        return [{"timestamp": 0, "visual": "[ERROR] Model captioning belum dimuat."}]
    if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
        print(f"[ERROR] File video tidak ditemukan atau kosong: {video_path}")
        return [{"timestamp": 0, "visual": "[ERROR] File video tidak ditemukan atau kosong."}]
    try:
        container = av.open(video_path)
        duration = float(container.duration / av.time_base)
        container.close()
        timestamps = [0, duration / 2, max(0, duration - 1)]
        for ts in timestamps:
            frame = extract_frame_at(video_path, ts)
            if frame:
                caption = caption_frame(frame)
                if not caption or caption.strip() == "":
                    print(f"[WARNING] Caption kosong di timestamp {ts}, isi dengan placeholder.")
                    caption = "[NO CAPTION]"
            else:
                caption = "[WARNING] Frame tidak ditemukan"
            results.append({
                "timestamp": ts,
                "visual": caption
            })
    except Exception as e:
        print(f"[ERROR] Gagal generate caption untuk {video_path}: {e}")
        results.append({"timestamp": 0, "visual": f"[ERROR] {str(e)}"})
    return results

# def _generate_single_caption(video_path: str):
#     if not PROCESSOR or not MODEL:
#         return []

#     # 1. Ambil audio & segmen
#     temp_audio = f"temp_{uuid.uuid4().hex}.wav"
#     audio_path = extract_audio(video_path, temp_audio)
#     if not audio_path:
#         return []

#     segments = transcribe_with_segments(audio_path)
#     os.remove(audio_path) if os.path.exists(audio_path) else None

#     results = []
#     for seg in segments:
#         mid_time = (seg["start"] + seg["end"]) / 2
#         frame = extract_frame_at(video_path, mid_time)
#         if frame:
#             caption = caption_frame(frame)
#         else:
#             caption = "[WARNING] Frame tidak ditemukan"
#         results.append({
#             "start": seg["start"],
#             "end": seg["end"],
#             "transcript": seg["text"],
#             "visual": caption
#         })
#     return results

# ================== Fungsi Utama ==================
def generate_captions_parallel(video_paths, max_workers=4):
    results = {}
    print(f"📂 Mulai proses {len(video_paths)} video...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_generate_single_caption, path): path for path in video_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                segments = future.result()
                results[path] = segments
                print(f"[SUCCESS] {os.path.basename(path)}")
            except Exception as e:
                results[path] = [{"error": str(e)}]
                print(f"[ERROR] {os.path.basename(path)}: {e}")
    return results

def caption_videos_in_folder(video_folder_path: str) -> pd.DataFrame:
    folder_path = Path(video_folder_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in folder_path.iterdir() if f.suffix.lower() in video_extensions]

    results = generate_captions_parallel(video_files, max_workers=4)
    rows = []
    for v, segs in results.items():
        for seg in segs:
            rows.append({
                "Filename": os.path.basename(v),
                "Start": seg.get("start"),
                "End": seg.get("end"),
                "Transcript": seg.get("transcript"),
                "Visual": seg.get("visual")
            })
    return pd.DataFrame(rows)
