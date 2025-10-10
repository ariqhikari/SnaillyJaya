import os
import re
import json
import hashlib
import logging
import subprocess
import pandas as pd
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def safe_filename(name: str) -> str:
    """Membersihkan nama file dari karakter ilegal agar aman digunakan di sistem file."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def download_video_ytdlp(url: str, output_folder: str) -> dict:
    """
    Mencoba download video menggunakan yt-dlp versi terbaru.
    Jika gagal, mengembalikan None agar fallback dijalankan.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "%(title)s.%(ext)s")

        # Try primary format, then fallback to a more flexible merged format
        cmds = [
            [
                "yt-dlp",
                "-f", "mp4",
                "-o", output_path,
                "--write-thumbnail",
                "--write-info-json",
                "--no-warnings",
                "--quiet",
                url
            ],
            [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio/best",
                "-o", output_path,
                "--merge-output-format", "mp4",
                "--write-thumbnail",
                "--write-info-json",
                "--no-warnings",
                "--quiet",
                url
            ]
        ]

        ran = False
        last_err = None
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=True)
                ran = True
                break
            except subprocess.CalledProcessError as e:
                last_err = e
                logging.debug(f"yt-dlp attempt failed with cmd: {cmd} -> {e}")

        if not ran:
            # Raise final error to be handled by caller
            raise last_err

        # Cari file info.json
        info_json = next(
            (f for f in os.listdir(output_folder) if f.endswith(".info.json")), None
        )
        if not info_json:
            raise FileNotFoundError("File info JSON tidak ditemukan.")

        with open(os.path.join(output_folder, info_json), "r", encoding="utf-8") as f:
            info = json.load(f)

        logging.info(f"✅ Berhasil download video: {info.get('title')}")
        return {
            "url": url,
            "title": info.get("title"),
            "description": info.get("description"),
            "thumbnail": info.get("thumbnail"),
            "file": os.path.join(output_folder, safe_filename(info.get("title")) + ".mp4")
        }

    except Exception as e:
        logging.warning(f"⚠️ Gagal download video via yt-dlp: {e}")
        return None


def fallback_scrape_with_webdriver(url: str, output_folder: str) -> dict:
    """
    Fallback menggunakan Selenium jika yt-dlp gagal.
    Mengambil thumbnail, judul, dan deskripsi.
    """
    logging.info("🔁 Menggunakan fallback WebDriver...")

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    data = {"url": url, "title": None, "description": None, "thumbnail": None}

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "title"))
        )

        data["title"] = driver.title
        desc_meta = driver.find_elements(By.CSS_SELECTOR, "meta[name='description']")
        if desc_meta:
            data["description"] = desc_meta[0].get_attribute("content")

        thumb_meta = driver.find_elements(By.CSS_SELECTOR, "link[rel='image_src'], meta[property='og:image']")
        if thumb_meta:
            data["thumbnail"] = (
                thumb_meta[0].get_attribute("href") or
                thumb_meta[0].get_attribute("content")
            )

        if data["thumbnail"]:
            os.makedirs(output_folder, exist_ok=True)
            img_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
            thumb_path = os.path.join(output_folder, f"thumbnail_{img_hash}.jpg")

            # Unduh thumbnail langsung lewat JavaScript fetch()
            img_data = driver.execute_script("""
                const url = arguments[0];
                return fetch(url)
                    .then(r => r.arrayBuffer())
                    .then(b => Array.from(new Uint8Array(b)));
            """, data["thumbnail"])

            if img_data:
                with open(thumb_path, "wb") as f:
                    f.write(bytearray(img_data))
                data["thumbnail_path"] = thumb_path
                # For compatibility with scraping pipeline, also expose image_folder and saved paths
                data["image_folder"] = output_folder
                data["saved_image_paths"] = [thumb_path]

        logging.info(f"📸 Berhasil ambil data fallback: {data['title']}")
        return data

    except Exception as e:
        logging.error(f"❌ Gagal scrape data via WebDriver: {e}")
        return data

    finally:
        driver.quit()


def scrape_video(url: str, base_folder: str = "downloads") -> dict:
    """
    Fungsi utama untuk scraping video:
    1. Coba download dengan yt-dlp.
    2. Jika gagal, fallback ke Selenium WebDriver.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_")
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    output_folder = os.path.join(base_folder, f"{domain}_{url_hash}")

    result = download_video_ytdlp(url, output_folder)
    if result:
        return result

    return fallback_scrape_with_webdriver(url, output_folder)



# --- Integrasi Captioning ---
from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
from src.utils.Preprocessing.VideoProcessor import caption_videos_in_folder

def scrape_to_dataframe(urls: list, base_folder: str = "downloads", caption_images: bool = True, caption_videos: bool = True) -> pd.DataFrame:
    """
    Scrape media sosial URLs, download media, and run captioning. Output matches Scrapping.py and ImageProcessor.py/VideoProcessor.py format.
    """
    results = []
    for url in urls:
        try:
            logging.info(f"🚀 Scraping {url}")
            data = scrape_video(url, base_folder=base_folder)
            # Tambahkan captioning untuk gambar
            image_folder = None
            video_folder = None
            image_caption_df = None
            video_caption_df = None
            # Normalize returned fields for downstream compatibility
            # Ensure we always have a 'text' field (title or description)
            if not data:
                data = {}
            if "title" in data and not data.get("title"):
                data["title"] = None
            # Use title as text if available
            data_text = data.get("text") or data.get("title") or data.get("description") or None
            data["text"] = data_text
            # Cek dan caption gambar
            # If yt-dlp downloaded a file, assume images are in same folder
            if caption_images and data.get("file") and os.path.exists(data["file"]):
                image_folder = os.path.dirname(data["file"]) 
                try:
                    image_caption_df = caption_images_in_folder(image_folder)
                except Exception as e:
                    logging.warning(f"Gagal captioning gambar: {e}")
            # Cek dan caption video
            if caption_videos and data.get("file") and data["file"].endswith(".mp4") and os.path.exists(data["file"]):
                video_folder = os.path.dirname(data["file"])
                try:
                    video_caption_df = caption_videos_in_folder(video_folder)
                except Exception as e:
                    logging.warning(f"Gagal captioning video: {e}")
            # If fallback scraping saved a thumbnail, surface it as image_folder
            if not image_folder and data.get("image_folder"):
                image_folder = data.get("image_folder")
                # do captioning on fallback image if requested
                if caption_images and data.get("saved_image_paths"):
                    try:
                        image_caption_df = caption_images_in_folder(image_folder)
                    except Exception as e:
                        logging.warning(f"Gagal captioning fallback images: {e}")
            # Gabungkan hasil
            result = {
                "url": url,
                "title": data.get("title"),
                "text": data.get("text"),
                "description": data.get("description"),
                "thumbnail": data.get("thumbnail"),
                "file": data.get("file"),
                "image_urls": ", ".join(data.get("saved_image_paths", [])) if data.get("saved_image_paths") else data.get("thumbnail"),
                "image_folder": image_folder,
                "saved_image_paths": data.get("saved_image_paths"),
                "video_folder": video_folder,
                "image_caption_df": image_caption_df,
                "video_caption_df": video_caption_df,
                "error": None
            }
            results.append(result)
        except Exception as e:
            logging.error(f"❌ Error scraping {url}: {e}")
            results.append({
                "url": url,
                "title": None,
                "description": None,
                "thumbnail": None,
                "file": None,
                "image_caption_df": None,
                "video_caption_df": None,
                "error": str(e)
            })

    # Buat DataFrame utama
    df = pd.DataFrame(results)
    output_path = os.path.join(base_folder, "scraping_summary.csv")
    os.makedirs(base_folder, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logging.info(f"✅ DataFrame hasil scraping disimpan ke {output_path}")
    return df
