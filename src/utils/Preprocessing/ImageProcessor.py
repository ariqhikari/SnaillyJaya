from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed

# --- Inisialisasi Model (Hanya dilakukan SEKALI saat modul di-load) ---
# Ini adalah bagian terpenting untuk efisiensi. Model yang besar hanya dimuat sekali.

print("Menginisialisasi komponen Image Captioning...")
# Izinkan memuat gambar yang mungkin 'rusak' atau terpotong
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Tentukan perangkat (GPU jika tersedia, jika tidak CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "public/models/blip_captioning"
# Muat model dan processor dari Hugging Face
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
        PROCESSOR.save_pretrained(MODEL_PATH)
        MODEL.save_pretrained(MODEL_PATH)
except Exception as e:
    print(f"GAGAL memuat model: {e}")
    print("Fungsi captioning tidak akan bekerja. Pastikan koneksi internet stabil dan library terinstal.")
    PROCESSOR = None
    MODEL = None

# --- Fungsi Bantuan ---

def _generate_single_caption(image_path: str):
    """
    Fungsi helper internal untuk membuat caption dari satu gambar.
    Menggunakan model dan processor global yang sudah dimuat.
    """
    try:
        print(f"Memproses gambar: {os.path.basename(image_path)}")
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[SKIP] Gagal membuka {os.path.basename(image_path)}: {e}")
        return "[ERROR] Gambar korup atau tidak dapat dibaca"

    try:
        # Proses gambar dan buat caption
        print(f"Memulai captioning untuk {os.path.basename(image_path)}...")
        inputs = PROCESSOR(images=img, return_tensors="pt").to(DEVICE)
        generated_ids = MODEL.generate(
            pixel_values=inputs.pixel_values,
            # decoding_method= "Beam search",
            temperature=1.0,
            length_penalty=1.0,
            repetition_penalty=1.5,
            min_length=1,
            num_beams=5,
            max_length=50,
        )
        caption = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
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
    if not MODEL:
        print("Error: Model tidak dimuat. Proses dibatalkan.")
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


