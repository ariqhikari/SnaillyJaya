from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed
import threading

# --- Global State Variables untuk Lazy Loading ---
PROCESSOR = None
MODEL = None
MODEL_LOADED = False
MODEL_LOADING = False
_model_lock = threading.Lock()

# Device dan path configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "public/models/blip_captioning"

# Izinkan memuat gambar yang mungkin 'rusak' atau terpotong
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image_captioning_model():
    """
    Lazy loading untuk model BLIP Image Captioning.
    Model hanya dimuat saat function ini dipanggil, bukan saat module diimport.
    
    Returns:
        tuple: (processor, model) atau (None, None) jika gagal
    """
    global PROCESSOR, MODEL, MODEL_LOADED, MODEL_LOADING
    
    # Jika model sudah dimuat, return langsung
    if MODEL_LOADED and PROCESSOR is not None and MODEL is not None:
        return PROCESSOR, MODEL
    
    # Thread-safe loading dengan lock
    with _model_lock:
        # Double-check setelah acquire lock
        if MODEL_LOADED and PROCESSOR is not None and MODEL is not None:
            return PROCESSOR, MODEL
        
        if MODEL_LOADING:
            print("⏳ Model sedang dimuat oleh thread lain, menunggu...")
            # Tunggu sampai loading selesai
            while MODEL_LOADING:
                pass
            return PROCESSOR, MODEL
        
        MODEL_LOADING = True
        print("🚀 Menginisialisasi komponen Image Captioning (Lazy Loading)...")
        
        try:
            if os.path.exists(MODEL_PATH):
                print("🔄 Memuat model dari lokal...")
                PROCESSOR = BlipProcessor.from_pretrained(MODEL_PATH)
                MODEL = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
            else:
                print("⬇️ Mengunduh model dari Hugging Face...")
                PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

                print("💾 Menyimpan model untuk penggunaan berikutnya...")
                os.makedirs(MODEL_PATH, exist_ok=True)
                PROCESSOR.save_pretrained(MODEL_PATH)
                MODEL.save_pretrained(MODEL_PATH)
            
            MODEL_LOADED = True
            print("✅ Model Image Captioning berhasil dimuat!")
            return PROCESSOR, MODEL
            
        except Exception as e:
            print(f"❌ GAGAL memuat model: {e}")
            print("⚠️ Fungsi captioning tidak akan bekerja. Pastikan koneksi internet stabil dan library terinstal.")
            PROCESSOR = None
            MODEL = None
            MODEL_LOADED = False
            return None, None
        finally:
            MODEL_LOADING = False

# --- Fungsi Bantuan ---

def _generate_single_caption(image_path: str):
    """
    Fungsi helper internal untuk membuat caption dari satu gambar.
    Menggunakan model dan processor global yang sudah dimuat dengan lazy loading.
    """
    # Lazy load model jika belum dimuat
    processor, model = load_image_captioning_model()
    
    if processor is None or model is None:
        print(f"[SKIP] Model tidak tersedia untuk {os.path.basename(image_path)}")
        return "[ERROR] Model tidak tersedia"
    
    try:
        print(f"Memproses gambar: {os.path.basename(image_path)}")
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[SKIP] Gagal membuka {os.path.basename(image_path)}: {e}")
        return "[ERROR] Gambar korup atau tidak dapat dibaca"

    try:
        # Proses gambar dan buat caption
        print(f"Memulai captioning untuk {os.path.basename(image_path)}...")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            # decoding_method= "Beam search",
            temperature=1.0,
            length_penalty=1.0,
            repetition_penalty=1.5,
            min_length=1,
            num_beams=5,
            max_length=50,
        )
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"Caption yang dihasilkan untuk {os.path.basename(image_path)}: {caption}")
        return caption
    except Exception as e:
        print(f"[SKIP] Gagal captioning {os.path.basename(image_path)}: {e}")
        return "[ERROR] Proses captioning gagal"

# --- Fungsi Utama ---
def generate_captions_parallel(image_paths, max_workers=4):
    """
    Memproses banyak gambar secara paralel untuk membuat caption.
    """
    results = {}
    print(f"Ditemukan {len(image_paths)} gambar. Memulai proses captioning...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print("Memulai proses captioning paralel...")
        future_to_path = {executor.submit(_generate_single_caption, path): path for path in image_paths}
        print(f"Future to path mapping: {future_to_path}")
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
                print(f"[SUCCESS] Caption untuk {path} berhasil dibuat.")
            except Exception as e:
                results[path] = f"[ERROR] {e}"
                print(f"[ERROR] Caption untuk {path} gagal dibuat.")
    return results

def caption_images_in_folder(
    image_folder_path: str,
    delete_corrupted: bool = False
) -> pd.DataFrame:
    """
    Membuat caption untuk semua gambar dalam sebuah folder dan mengembalikan hasilnya
    sebagai Pandas DataFrame.

    Args:
        image_folder_path (str): Path lengkap menuju folder yang berisi gambar.
        delete_corrupted (bool, optional): Jika True, gambar yang gagal diproses(korup/error) akan dihapus. Defaultnya adalah False.

    Returns:
        pd.DataFrame: Sebuah DataFrame dengan kolom "Filename" dan "Caption". Mengembalikan DataFrame kosong jika terjadi error atau tidak ada gambar.
    """
    # Lazy load model sebelum memproses
    processor, model = load_image_captioning_model()
    
    if processor is None or model is None:
        print("❌ Error: Model tidak dimuat. Proses dibatalkan.")
        return pd.DataFrame()

    folder_path = Path(image_folder_path)
    if not folder_path.is_dir():
        print(f"Error: Folder tidak ditemukan di '{image_folder_path}'")
        return pd.DataFrame()

    # Cari file gambar yang valid
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    if not image_files:
        print(f"Tidak ada gambar yang ditemukan di folder '{image_folder_path}'")
        return pd.DataFrame()
    print(f"Ditemukan {len(image_files)} gambar di folder '{image_folder_path}'")
    captions = generate_captions_parallel(image_files, max_workers=10)
    print(f"Ditemukan {len(captions)} gambar. Memulai proses captioning...")
    print(captions)

    print("Proses captioning selesai.")
    print("Proses captioning selesai.")
    df = pd.DataFrame({"Caption": list(captions.values())})
    return df


# --- Fungsi Publik untuk Caption Satu Gambar ---
def caption_image(image_path: str) -> str:
    """
    Membuat caption untuk satu gambar (wrapper publik).
    """
    return _generate_single_caption(image_path)
