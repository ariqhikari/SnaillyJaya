
# ================== IMPORTS ==================
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import yt_dlp
import av
from pathlib import Path

# ================== MODEL SETUP ==================
print("🚀 Menginisialisasi model captioning...")
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_NAME = "microsoft/git-base-coco"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model berhasil diinisialisasi ({device.upper()})")

# ================== YOUTUBE DOWNLOADER ==================
class YouTubeDownloader:
    """Downloader video YouTube menggunakan yt-dlp."""
    def __init__(self, download_folder="downloads"):
        self.download_folder = Path(download_folder)
        self.download_folder.mkdir(exist_ok=True)

    def download_video(self, url):
        """Coba download video, jika gagal → fallback ambil metadata."""
        print(f"\n🎥 Memproses URL: {url}")
        ydl_opts = {
            "format": "mp4/bestaudio/best",
            "outtmpl": str(self.download_folder / "%(id)s.%(ext)s"),
            "quiet": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"✅ Video berhasil diunduh: {filename}")
                return filename
        except Exception as e:
            print(f"⚠️ Gagal mengunduh video: {e}")
            print("➡️ Mengambil metadata sebagai fallback...")
            metadata = self.extract_video_metadata(url)
            return metadata if metadata else None

    def extract_video_metadata(self, url):
        """Ambil title, description, dan thumbnail tanpa download video."""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown Title"),
                    "description": info.get("description", "No Description"),
                    "thumbnail": info.get("thumbnail", None),
                    "url": url,
                }
        except Exception as e:
            print(f"❌ Gagal ambil metadata: {e}")
            return None

# ================== FRAME EXTRACTION ==================
def extract_frames(video_path, num_frames=3):
    """Ambil frame di awal, tengah, dan akhir video."""
    frames = []
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        frame_positions = [
            int(total_frames * 0.1),
            int(total_frames * 0.5),
            int(total_frames * 0.9),
        ]
        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_positions:
                img = frame.to_image()
                frames.append(img)
            if len(frames) == num_frames:
                break
        container.close()
    except Exception as e:
        print(f"⚠️ Gagal ekstraksi frame: {e}")
    return frames

# ================== CAPTION GENERATION ==================
def generate_caption(image: Image.Image):
    """Menghasilkan caption dari satu frame."""
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    except Exception as e:
        print(f"⚠️ Gagal generate caption: {e}")
        return "[Caption gagal dihasilkan]"

# ================== MAIN FUNCTIONS ==================
def caption_videos_in_folder(video_folder_path: str, delete_corrupted: bool=False) -> pd.DataFrame:
    """
    Membuat caption untuk semua video dalam sebuah folder dan mengembalikan hasilnya sebagai DataFrame.
    Args:
        video_folder_path (str): Path folder video.
        delete_corrupted (bool): Hapus video korup jika True.
    Returns:
        pd.DataFrame: Kolom Filename, Caption, Video_Path.
    """
    folder_path = Path(video_folder_path)
    if not folder_path.is_dir():
        print(f"❌ Folder '{video_folder_path}' tidak ditemukan")
        return pd.DataFrame()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    if not video_files:
        print(f"⚠️ Tidak ada video ditemukan di '{video_folder_path}'")
        return pd.DataFrame()
    rows = []
    for v in video_files:
        try:
            frames = extract_frames(str(v))
            if not frames:
                caption = "[ERROR] Tidak ada frame valid"
            else:
                captions = [generate_caption(f) for f in frames]
                caption = " ".join(captions)
            rows.append({
                "Filename": v.name,
                "Caption": caption,
                "Video_Path": str(v)
            })
        except Exception as e:
            rows.append({
                "Filename": v.name,
                "Caption": f"[ERROR] {e}",
                "Video_Path": str(v)
            })
    return pd.DataFrame(rows)

def download_and_caption_youtube_video(url, output_folder="output_vid"):
    """
    Unduh video dari YouTube dan hasilkan caption. Jika gagal, ambil metadata.
    Returns DataFrame dengan kolom standar.
    """
    downloader = YouTubeDownloader()
    Path(output_folder).mkdir(exist_ok=True)
    result_rows = []
    video_data = downloader.download_video(url)
    if isinstance(video_data, dict):
        print("📄 Menyimpan hasil fallback metadata...")
        result_rows.append({
            "Filename": "[VIDEO NOT DOWNLOADED]",
            "Timestamp": 0,
            "Caption": "[NO VIDEO - ONLY METADATA]",
            "Position": "METADATA",
            "Video_Path": None,
            "Title": video_data["title"],
            "Description": video_data["description"],
            "Thumbnail": video_data["thumbnail"],
            "YouTube_URL": video_data["url"],
        })
    elif isinstance(video_data, str) and os.path.exists(video_data):
        frames = extract_frames(video_data)
        if not frames:
            print("⚠️ Tidak ada frame diambil dari video.")
        else:
            print("🧠 Menghasilkan caption untuk setiap frame...")
            positions = ["AWAL", "TENGAH", "AKHIR"]
            for idx, (frame, pos) in enumerate(zip(frames, positions)):
                caption = generate_caption(frame)
                result_rows.append({
                    "Filename": os.path.basename(video_data),
                    "Timestamp": idx,
                    "Caption": caption,
                    "Position": pos,
                    "Video_Path": video_data,
                    "Title": "",
                    "Description": "",
                    "Thumbnail": "",
                    "YouTube_URL": url,
                })
    else:
        print("❌ Tidak ada hasil yang dapat disimpan.")
    df = pd.DataFrame(result_rows)
    csv_path = Path(output_folder) / "caption_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Hasil disimpan ke: {csv_path}")
    return df

# ===============================================================
# MODEL SETUP
# ===============================================================

print("🚀 Menginisialisasi model captioning...")
ImageFile.LOAD_TRUNCATED_IMAGES = True

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model berhasil diinisialisasi ({device.upper()})")

# ===============================================================
# CLASS: YouTube Downloader (yt-dlp)
# ===============================================================

class YouTubeDownloader:
    def __init__(self, download_folder="downloads"):
        self.download_folder = Path(download_folder)
        self.download_folder.mkdir(exist_ok=True)

    def download_video(self, url):
        """Coba download video, jika gagal → fallback ambil metadata"""
        print(f"\n🎥 Memproses URL: {url}")
        ydl_opts = {
            "format": "mp4/bestaudio/best",
            "outtmpl": str(self.download_folder / "%(id)s.%(ext)s"),
            "quiet": True,
            "noplaylist": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"✅ Video berhasil diunduh: {filename}")
                return filename
        except Exception as e:
            print(f"⚠️ Gagal mengunduh video: {e}")
            print("➡️ Mengambil metadata sebagai fallback...")
            metadata = self.extract_video_metadata(url)
            if metadata:
                return metadata
            else:
                print("❌ Gagal mengambil metadata juga.")
                return None

    def extract_video_metadata(self, url):
        """Ambil title, description, dan thumbnail tanpa download video"""
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", "Unknown Title"),
                    "description": info.get("description", "No Description"),
                    "thumbnail": info.get("thumbnail", None),
                    "url": url,
                }
        except Exception as e:
            print(f"❌ Gagal ambil metadata: {e}")
            return None

# ===============================================================
# FUNCTION: Ekstraksi Frame dari Video
# ===============================================================

def extract_frames(video_path, num_frames=3):
    """Ambil frame di awal, tengah, dan akhir video"""
    frames = []
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        frame_positions = [
            int(total_frames * 0.1),
            int(total_frames * 0.5),
            int(total_frames * 0.9),
        ]

        for i, frame in enumerate(container.decode(video=0)):
            if i in frame_positions:
                img = frame.to_image()
                frames.append(img)
            if len(frames) == num_frames:
                break
        container.close()
    except Exception as e:
        print(f"⚠️ Gagal ekstraksi frame: {e}")
    return frames

# ===============================================================
# FUNCTION: Generate Caption
# ===============================================================

def generate_caption(image: Image.Image):
    """Menghasilkan caption dari satu frame"""
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()
    except Exception as e:
        print(f"⚠️ Gagal generate caption: {e}")
        return "[Caption gagal dihasilkan]"

# ===============================================================
# MAIN FUNCTION: Download + Captioning
# ===============================================================

def download_and_caption_youtube_video(url, output_folder="output_vid"):
    """
    Unduh video dari YouTube dan hasilkan caption.
    Jika gagal download, ambil metadata (title, desc, thumbnail).
    """

    downloader = YouTubeDownloader()
    Path(output_folder).mkdir(exist_ok=True)
    result_rows = []

    video_data = downloader.download_video(url)

    # CASE 1 → Metadata fallback (bukan path file)
    if isinstance(video_data, dict):
        print("📄 Menyimpan hasil fallback metadata...")
        result_rows.append({
            "Filename": "[VIDEO NOT DOWNLOADED]",
            "Timestamp": 0,
            "Caption": "[NO VIDEO - ONLY METADATA]",
            "Position": "METADATA",
            "Video_Path": None,
            "Title": video_data["title"],
            "Description": video_data["description"],
            "Thumbnail": video_data["thumbnail"],
            "YouTube_URL": video_data["url"],
        })

    # CASE 2 → Video berhasil diunduh
    elif isinstance(video_data, str) and os.path.exists(video_data):
        frames = extract_frames(video_data)
        if not frames:
            print("⚠️ Tidak ada frame diambil dari video.")
        else:
            print("🧠 Menghasilkan caption untuk setiap frame...")
            positions = ["AWAL", "TENGAH", "AKHIR"]
            for idx, (frame, pos) in enumerate(zip(frames, positions)):
                caption = generate_caption(frame)
                result_rows.append({
                    "Filename": os.path.basename(video_data),
                    "Timestamp": idx,  # Optionally use frame index or timestamp
                    "Caption": caption,
                    "Position": pos,
                    "Video_Path": video_data,
                    "Title": "",
                    "Description": "",
                    "Thumbnail": "",
                    "YouTube_URL": url,
                })

    else:
        print("❌ Tidak ada hasil yang dapat disimpan.")

    # Simpan ke CSV
    df = pd.DataFrame(result_rows)
    csv_path = Path(output_folder) / "caption_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Hasil disimpan ke: {csv_path}")

    return df

# from concurrent.futures import ThreadPoolExecutor, as_completed
# import os
# import pandas as pd
# import torch
# from PIL import Image, ImageFile
# from transformers import AutoProcessor, AutoModelForCausalLM
# from tqdm import tqdm
# import av
# from pathlib import Path
# import yt_dlp

# """
# Video Captioning Service - Visual Only dengan YouTube Downloader
# Hanya menggunakan GIT model untuk captioning frame video
# """

# print("🚀 Menginisialisasi komponen Video Captioning...")
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ================== Model Captioning ==================
# GIT_MODEL_PATH = "public/models/VIDgit_captioning"
# PROCESSOR, MODEL = None, None

# def load_git_model():
#     """Load GIT model untuk visual captioning"""
#     global PROCESSOR, MODEL
#     try:
#         if os.path.exists(GIT_MODEL_PATH):
#             print("🔄 Memuat GIT model dari lokal...")
#             PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_PATH)
#             MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_PATH).to(DEVICE)
#         else:
#             print("⬇️ Mengunduh GIT model dari Hugging Face...")
#             PROCESSOR = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
#             MODEL = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex").to(DEVICE)
#             os.makedirs(GIT_MODEL_PATH, exist_ok=True)
#             PROCESSOR.save_pretrained(GIT_MODEL_PATH)
#             MODEL.save_pretrained(GIT_MODEL_PATH)
#         print("✅ GIT model berhasil dimuat")
#     except Exception as e:
#         print(f"❌ Gagal memuat GIT model: {e}")
#         raise

# # Load model saat import
# load_git_model()

# # ================== YouTube Downloader ==================
# class YouTubeDownloader:
#     def __init__(self, download_dir="downloads"):
#         self.download_dir = Path(download_dir)
#         self.download_dir.mkdir(exist_ok=True)
        
#     def download_video(self, url):
#         """
#         Download video YouTube dengan format yang lebih fleksibel
#         """
#         try:
#             # Buat folder khusus untuk video ini
#             video_id = url.split('v=')[-1].split('&')[0]
#             video_folder = self.download_dir / video_id
#             video_folder.mkdir(exist_ok=True)
            
#             ydl_opts = {
#                 'outtmpl': str(video_folder / '%(title)s.%(ext)s'),
#                 'format': 'best[height<=480]/best[height<=720]/best',  # Format fleksibel
#                 'quiet': False,
#                 'no_warnings': False,
#             }
            
#             print(f"⬇️ Mendownload video dari: {url}")
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 info = ydl.extract_info(url, download=True)
#                 video_path = ydl.prepare_filename(info)
                
#                 # Pastikan file ada
#                 if os.path.exists(video_path):
#                     print(f"✅ Video berhasil didownload: {video_path}")
#                     print(f"📊 Video info: {info.get('title', 'Unknown')} - {info.get('duration', 0)}s")
#                     return str(video_path), str(video_folder)
#                 else:
#                     print(f"❌ File video tidak ditemukan setelah download")
#                     return None, None
                    
#         except Exception as e:
#             print(f"❌ Error downloading video: {e}")
#             # Coba format fallback yang lebih sederhana
#             return self.download_video_fallback(url)

#     def download_video_fallback(self, url):
#         """Fallback method dengan format yang lebih sederhana"""
#         try:
#             video_id = url.split('v=')[-1].split('&')[0]
#             video_folder = self.download_dir / video_id
#             video_folder.mkdir(exist_ok=True)
            
#             ydl_opts = {
#                 'outtmpl': str(video_folder / '%(title)s.%(ext)s'),
#                 'format': 'best',  # Ambil format terbaik yang available
#                 'quiet': False,
#             }
            
#             print(f"🔄 Mencoba download dengan format fallback...")
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 info = ydl.extract_info(url, download=True)
#                 video_path = ydl.prepare_filename(info)
                
#                 if os.path.exists(video_path):
#                     print(f"✅ Video berhasil didownload (fallback): {video_path}")
#                     return str(video_path), str(video_folder)
#                 else:
#                     return None, None
                    
#         except Exception as e:
#             print(f"❌ Fallback download juga gagal: {e}")
#             return None, None

# # ================== Fungsi Video Processing ==================

# def extract_frame_at(video_path, timestamp):
#     """Ambil 1 frame dari video pada detik tertentu"""
#     try:
#         container = av.open(video_path)
#         stream = container.streams.video[0]
#         # Convert timestamp to the stream's time base
#         target_time = int(timestamp * stream.time_base.denominator)
#         container.seek(target_time)
        
#         for frame in container.decode(video=0):
#             img = frame.to_image().convert("RGB")
#             container.close()
#             return img
#         container.close()
#     except Exception as e:
#         print(f"[ERROR] Gagal ambil frame di {timestamp}s: {e}")
#     return None

# def caption_frame(image):
#     """Generate caption untuk single frame"""
#     if PROCESSOR is None or MODEL is None:
#         return "[ERROR] Model captioning tidak tersedia"
    
#     try:
#         inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
#         generated_ids = MODEL.generate(
#             pixel_values=inputs.pixel_values,
#             temperature=1.0,
#             repetition_penalty=1.5,
#             num_beams=3,
#             max_length=50,
#         )
#         caption = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#         return caption if caption else "[NO CAPTION]"
#     except Exception as e:
#         return f"[ERROR] Caption gagal: {e}"

# def get_video_duration(video_path):
#     """Dapatkan durasi video"""
#     try:
#         container = av.open(video_path)
#         duration = float(container.duration / av.time_base)
#         container.close()
#         return duration
#     except Exception as e:
#         print(f"[ERROR] Gagal mendapatkan durasi video: {e}")
#         return 0

# def _generate_single_caption(video_path: str):
#     """Generate captions untuk satu video (3 frame: AWAL, TENGAH, AKHIR)"""
#     results = []
    
#     # Validasi model
#     if not PROCESSOR or not MODEL:
#         return [{
#             "timestamp": 0, 
#             "visual": "[ERROR] Model captioning belum dimuat.",
#             "filename": os.path.basename(video_path)
#         }]
    
#     # Validasi file
#     if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
#         return [{
#             "timestamp": 0, 
#             "visual": "[ERROR] File video tidak ditemukan atau kosong.",
#             "filename": os.path.basename(video_path)
#         }]
    
#     try:
#         # Dapatkan durasi video
#         duration = get_video_duration(video_path)
#         if duration == 0:
#             return [{
#                 "timestamp": 0, 
#                 "visual": "[ERROR] Tidak bisa membaca durasi video",
#                 "filename": os.path.basename(video_path)
#             }]
        
#         # Tentukan timestamps: AWAL, TENGAH, AKHIR
#         timestamps = [
#             0,                          # Frame AWAL
#             duration / 2,               # Frame TENGAH  
#             max(0, duration - 1)        # Frame AKHIR (1 detik sebelum selesai)
#         ]
        
#         print(f"🎬 Processing {os.path.basename(video_path)}")
#         print(f"   📏 Durasi: {duration:.2f}s")
#         print(f"   ⏱️  Timestamps: {[f'{ts:.1f}s' for ts in timestamps]}")
        
#         # Process setiap timestamp
#         for i, ts in enumerate(timestamps):
#             position = ["AWAL", "TENGAH", "AKHIR"][i]
#             print(f"   🖼️  Mengambil frame {position} ({ts:.1f}s)...")
            
#             frame = extract_frame_at(video_path, ts)
#             if frame:
#                 print(f"   🤖 Generating caption untuk frame {position}...")
#                 caption = caption_frame(frame)
#                 # Handle empty caption
#                 if not caption or caption.strip() == "":
#                     caption = "[NO CAPTION]"
#                 print(f"   ✅ Frame {position}: {caption}")
#             else:
#                 caption = "[ERROR] Frame tidak bisa diekstrak"
#                 print(f"   ❌ Gagal extract frame {position}")
            
#             results.append({
#                 "timestamp": ts,
#                 "visual": caption,
#                 "filename": os.path.basename(video_path),
#                 "position": position
#             })
            
#     except Exception as e:
#         print(f"[ERROR] Gagal generate caption untuk {video_path}: {e}")
#         results.append({
#             "timestamp": 0, 
#             "visual": f"[ERROR] {str(e)}",
#             "filename": os.path.basename(video_path),
#             "position": "ERROR"
#         })
    
#     return results

# def generate_captions_parallel(video_paths, max_workers=2):
#     """Process multiple videos in parallel"""
#     results = {}
#     print(f"📂 Mulai proses {len(video_paths)} video...")
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_path = {
#             executor.submit(_generate_single_caption, path): path 
#             for path in video_paths
#         }
        
#         for future in tqdm(as_completed(future_to_path), total=len(video_paths), desc="Processing videos"):
#             path = future_to_path[future]
#             try:
#                 segments = future.result()
#                 results[path] = segments
#                 print(f"✅ [SUCCESS] {os.path.basename(path)} - {len(segments)} captions")
#             except Exception as e:
#                 results[path] = [{
#                     "timestamp": 0, 
#                     "visual": f"[ERROR] {str(e)}",
#                     "filename": os.path.basename(path),
#                     "position": "ERROR"
#                 }]
#                 print(f"❌ [ERROR] {os.path.basename(path)}: {e}")
    
#     return results

# def caption_videos_in_folder(video_folder_path: str) -> pd.DataFrame:
#     """Main function untuk captioning semua video dalam folder"""
#     folder_path = Path(video_folder_path)
    
#     # Validasi folder
#     if not folder_path.exists():
#         raise FileNotFoundError(f"Folder tidak ditemukan: {video_folder_path}")
    
#     # Cari file video
#     video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
#     video_files = [
#         str(f) for f in folder_path.iterdir() 
#         if f.is_file() and f.suffix.lower() in video_extensions
#     ]
    
#     if not video_files:
#         print(f"⚠️ Tidak ada file video yang ditemukan di: {video_folder_path}")
#         return pd.DataFrame()
    
#     print(f"🎬 Found {len(video_files)} video files")
    
#     # Process videos
#     results = generate_captions_parallel(video_files, max_workers=2)
    
#     # Convert to DataFrame
#     rows = []
#     for video_path, segments in results.items():
#         for seg in segments:
#             rows.append({
#                 "Filename": seg.get("filename", os.path.basename(video_path)),
#                 "Timestamp": seg.get("timestamp", 0),
#                 "Caption": seg.get("visual", "[NO CAPTION]"),
#                 "Position": seg.get("position", "UNKNOWN"),
#                 "Video_Path": video_path
#             })
    
#     df = pd.DataFrame(rows)
    
#     # Summary
#     success_count = len([cap for cap in df['Caption'] if not cap.startswith('[ERROR]') and not cap.startswith('[NO CAPTION]')])
#     print(f"📊 Summary: {success_count}/{len(df)} captions berhasil dihasilkan")
    
#     # Print detail hasil
#     print("\n🎯 DETAIL HASIL CAPTIONING:")
#     for _, row in df.iterrows():
#         print(f"   📍 {row['Position']} ({row['Timestamp']:.1f}s): {row['Caption']}")
    
#     return df

# def download_and_caption_youtube_video(youtube_url: str) -> pd.DataFrame:
#     """
#     Fungsi utama: Download video YouTube dan generate captions untuk 3 frame
#     """
#     print(f"🎬 MEMULAI PROCESSING YOUTUBE URL: {youtube_url}")
    
#     # Download video
#     downloader = YouTubeDownloader()
#     video_path, video_folder = downloader.download_video(youtube_url)
    
#     if video_path and video_folder:
#         print(f"✅ Video berhasil didownload: {video_path}")
        
#         # Cek apakah file benar-benar ada
#         if os.path.exists(video_path):
#             file_size = os.path.getsize(video_path)
#             print(f"📄 File exists: Yes, size: {file_size} bytes")
#         else:
#             print(f"❌ File exists: No")
#             return pd.DataFrame()
        
#         # Process video untuk dapatkan caption (AWAL, TENGAH, AKHIR)
#         print("🔄 Memulai proses captioning 3 frame (AWAL, TENGAH, AKHIR)...")
#         df_captions = caption_videos_in_folder(video_folder)
        
#         if not df_captions.empty:
#             print(f"🎉 BERHASIL! Generated {len(df_captions)} captions dari video YouTube")
            
#             # Tambahkan kolom YouTube URL ke DataFrame
#             df_captions['YouTube_URL'] = youtube_url
            
#             # Gabungkan semua caption menjadi satu teks
#             all_captions = ". ".join(df_captions['Caption'].tolist())
#             print(f"📝 ALL CAPTIONS COMBINED: {all_captions}")
            
#             return df_captions
#         else:
#             print("⚠️ Tidak ada caption yang dihasilkan dari video YouTube")
#             return pd.DataFrame()
#     else:
#         print("❌ Gagal download video YouTube")
#         return pd.DataFrame()

# # Contoh penggunaan
# # if __name__ == "__main__":
# #     # Test dengan folder video lokal
# #     test_folder = "path/to/your/video/folder"
# #     df_result = caption_videos_in_folder(test_folder)
# #     print(df_result.head())
    
# #     # Test dengan YouTube URL
# #     youtube_url = "https://www.youtube.com/watch?v=9MlpjTBb8Vg"
# #     df_youtube = download_and_caption_youtube_video(youtube_url)
# #     print(df_youtube.head())

# # from concurrent.futures import ThreadPoolExecutor, as_completed
# # import os
# # import pandas as pd
# # import torch
# # from PIL import Image, ImageFile
# # from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
# # from tqdm import tqdm
# # import av
# # import librosa
# # from pathlib import Path
# # import uuid

# # """
# # Hal yang harus diperharui:
# # 1. Penanganan error dikarenakan captioning tidak berhasil dilakukan
# # 2. Hanya mengambil judul video saja
# # 3. Gambar tidak masuk kedalam proses scrapping yang dimana itu menjadikan proses nomo 1 tidak berjalan
# # """

# # print("🚀 Menginisialisasi komponen Video Captioning...")
# # ImageFile.LOAD_TRUNCATED_IMAGES = True
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # ================== Model Captioning ==================
# # GIT_MODEL_PATH = "public/models/VIDgit_captioning"
# # PROCESSOR, MODEL = None, None
# # try:
# #     if os.path.exists(GIT_MODEL_PATH):
# #         print("🔄 Memuat GIT model dari lokal...")
# #         PROCESSOR = AutoProcessor.from_pretrained(GIT_MODEL_PATH)
# #         MODEL = AutoModelForCausalLM.from_pretrained(GIT_MODEL_PATH).to(DEVICE)
# #     else:
# #         print("⬇️ Mengunduh GIT model dari Hugging Face...")
# #         PROCESSOR = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
# #         MODEL = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex").to(DEVICE)
# #         PROCESSOR.save_pretrained(GIT_MODEL_PATH)
# #         MODEL.save_pretrained(GIT_MODEL_PATH)
# # except Exception as e:
# #     print(f"❌ Gagal memuat GIT model: {e}")

# # # ================== Model Speech to Text (Whisper) ==================
# # """
# # Pertanyaan yang terjadi untuk saaat ini:
# # 1. Sudah dijalankan kode ini, akan tetapi hasilya tidak terlihat
# # 2.
# # """
# # WHISPER_MODEL_PATH = "public/models/STT_captioning"
# # ASR = None
# # try:
# #     if os.path.exists(WHISPER_MODEL_PATH):
# #         print("🔄 Memuat Whisper dari lokal...")
# #         ASR = pipeline("automatic-speech-recognition", model=WHISPER_MODEL_PATH,
# #                        device=0 if DEVICE.type == "cuda" else -1)
# #     else:
# #         print("⬇️ Mengunduh Whisper model...")
# #         ASR = pipeline("automatic-speech-recognition", model="openai/whisper-medium",
# #                        device=0 if DEVICE.type == "cuda" else -1)
# #         os.makedirs(WHISPER_MODEL_PATH, exist_ok=True)
# #         ASR.model.save_pretrained(WHISPER_MODEL_PATH)
# #         ASR.tokenizer.save_pretrained(WHISPER_MODEL_PATH)
# # except Exception as e:
# #     print(f"❌ Gagal memuat Whisper model: {e}")

# # # ================== Fungsi Audio ==================
# # def extract_audio(video_path: str, audio_out: str) -> str:
# #     try:
# #         container = av.open(video_path)
# #         audio_streams = [s for s in container.streams if s.type == "audio"]
# #         if not audio_streams:
# #             return ""
# #         audio_stream = audio_streams[0]

# #         out = av.open(audio_out, 'w')
# #         out.add_stream(template=audio_stream)
# #         for packet in container.demux(audio_stream):
# #             out.mux(packet)
# #         out.close()
# #         container.close()
# #         return audio_out
# #     except Exception as e:
# #         print(f"[ERROR] Ekstrak audio gagal: {e}")
# #         return ""

# # def transcribe_with_segments(audio_path: str):
# #     """Transkripsi audio + ambil timestamp segmen (fallback kalau tidak ada chunks)"""
# #     if not ASR:
# #         return [] #ASR nya belum berjalan
# #     # Konversi ke WAV (16kHz mono) kalau belum
# #     if not audio_path.lower().endswith(".wav"):
# #         wav_path = os.path.splitext(audio_path)[0] + ".wav"
# #         new_wav = extract_audio(audio_path, wav_path)
# #         audio_path = new_wav if new_wav else audio_path

# #     # Load audio
# #     speech, sr = librosa.load(audio_path, sr=16000)

# #     # Jalankan ASR
# #     result = ASR({"array": speech, "sampling_rate": sr})

# #     chunks = []
# #     # Kalau hasil punya segmen (chunks)
# #     if isinstance(result, dict) and "chunks" in result:
# #         for seg in result["chunks"]:
# #             start, end = seg.get("timestamp", (None, None))
# #             text = seg.get("text", "").strip()
# #             if start is not None and end is not None and text:
# #                 chunks.append({
# #                     "start": float(start),
# #                     "end": float(end),
# #                     "text": text
# #                 })
# #     else:
# #         # Fallback → kalau tidak ada segmen, tetap simpan full text
# #         chunks.append({
# #             "start": 0.0,
# #             "end": len(speech) / sr,
# #             "text": result["text"].strip() if isinstance(result, dict) else str(result)
# #         })

# #     return chunks

# # # ================== Fungsi Video ==================
# # def extract_frame_at(video_path, timestamp):
# #     """Ambil 1 frame dari video pada detik tertentu"""
# #     try:
# #         container = av.open(video_path)
# #         stream = container.streams.video[0]
# #         container.seek(int(timestamp * stream.time_base.denominator))
# #         for frame in container.decode(video=0):
# #             img = frame.to_image().convert("RGB")
# #             container.close()
# #             return img
# #         container.close()
# #     except Exception as e:
# #         print(f"[ERROR] Gagal ambil frame di {timestamp}s: {e}")
# #     return None

# # def caption_frame(image):
# #     try:
# #         inputs = PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
# #         generated_ids = MODEL.generate(
# #             pixel_values=inputs.pixel_values,
# #             temperature=1.0,
# #             repetition_penalty=1.5,
# #             num_beams=3,
# #             max_length=50,
# #         )
# #         return PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# #     except Exception as e:
# #         return f"[ERROR] Caption gagal: {e}"

# # def _generate_single_caption(video_path: str):
# #     results = []
# #     if not PROCESSOR or not MODEL:
# #         print("[ERROR] Model captioning belum dimuat.")
# #         return [{"timestamp": 0, "visual": "[ERROR] Model captioning belum dimuat."}]
# #     if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
# #         print(f"[ERROR] File video tidak ditemukan atau kosong: {video_path}")
# #         return [{"timestamp": 0, "visual": "[ERROR] File video tidak ditemukan atau kosong."}]
# #     try:
# #         container = av.open(video_path)
# #         duration = float(container.duration / av.time_base)
# #         container.close()
# #         timestamps = [0, duration / 2, max(0, duration - 1)]
# #         for ts in timestamps:
# #             frame = extract_frame_at(video_path, ts)
# #             if frame:
# #                 caption = caption_frame(frame)
# #                 if not caption or caption.strip() == "":
# #                     print(f"[WARNING] Caption kosong di timestamp {ts}, isi dengan placeholder.")
# #                     caption = "[NO CAPTION]"
# #             else:
# #                 caption = "[WARNING] Frame tidak ditemukan"
# #             results.append({
# #                 "timestamp": ts,
# #                 "visual": caption
# #             })
# #     except Exception as e:
# #         print(f"[ERROR] Gagal generate caption untuk {video_path}: {e}")
# #         results.append({"timestamp": 0, "visual": f"[ERROR] {str(e)}"})
# #     return results

# # # ================== Fungsi Utama ==================
# # def generate_captions_parallel(video_paths, max_workers=4):
# #     results = {}
# #     print(f"📂 Mulai proses {len(video_paths)} video...")
# #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         future_to_path = {executor.submit(_generate_single_caption, path): path for path in video_paths}
# #         for future in as_completed(future_to_path):
# #             path = future_to_path[future]
# #             try:
# #                 segments = future.result()
# #                 results[path] = segments
# #                 print(f"[SUCCESS] {os.path.basename(path)}")
# #             except Exception as e:
# #                 results[path] = [{"error": str(e)}]
# #                 print(f"[ERROR] {os.path.basename(path)}: {e}")
# #     return results

# # def caption_videos_in_folder(video_folder_path: str) -> pd.DataFrame:
# #     folder_path = Path(video_folder_path)
# #     video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
# #     video_files = [f for f in folder_path.iterdir() if f.suffix.lower() in video_extensions]

# #     results = generate_captions_parallel(video_files, max_workers=4)
# #     rows = []
# #     for v, segs in results.items():
# #         for seg in segs:
# #             rows.append({
# #                 "Filename": os.path.basename(v),
# #                 "Start": seg.get("start"),
# #                 "End": seg.get("end"),
# #                 "Transcript": seg.get("transcript"),
# #                 "Visual": seg.get("visual")
# #             })
# #     return pd.DataFrame(rows)