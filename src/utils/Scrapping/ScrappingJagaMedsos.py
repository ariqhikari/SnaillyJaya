# social_scrapers.py
import os
import re
import time
import json
import requests
import hashlib
import pandas as pd
from urllib.parse import urljoin, urlparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import yt_dlp
try:
    from pytube import YouTube
except ImportError:
    YouTube = None

try:
    import instaloader
except Exception:
    instaloader = None

try:
    from googleapiclient.discovery import build as gbuild
except Exception:
    gbuild = None

# --- Helper Functions (Fungsi Bantuan) ---
def init_driver(headless=True):
    """Menginisialisasi dan mengembalikan instance WebDriver Selenium."""
    print("Menginisialisasi WebDriver...")
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--window-size=1920,1080")
    # Opsi tambahan untuk mengatasi beberapa proteksi dan error
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        print("WebDriver berhasil diinisialisasi.")
        return driver
    except ValueError as e:
        print(f"Error: Terjadi masalah saat menginstal ChromeDriver. Cek koneksi internet Anda. Detail: {e}")
        return None

# -------------------- Helpers: save images/media --------------------
import requests
def save_binary(url, folder, filename, headers=None):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    try:
        resp = requests.get(url, stream=True, timeout=20, headers=headers or {'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        return path
    except Exception as e:
        print(f"[WARN] Gagal download {url}: {e}")
        return None

def make_id_from_url(url):
    return hashlib.md5(url.encode('utf-8')).hexdigest()[:10]

# ============================================================
# Scraping per platform
# ============================================================
def scrape_youtube(driver, url):
    """Scrape metadata YouTube"""
    driver.get(url)
    wait = WebDriverWait(driver, 15)

    try:
        title = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "h1.title yt-formatted-string"))
        ).text
    except:
        title = "Unknown"

    try:
        views = driver.find_element(By.CSS_SELECTOR, ".view-count").text
    except:
        views = "Unknown"

    # Ambil video url dari halaman (jika ada)
    video_urls = []
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for vid in video_elements:
            src = vid.get_attribute("src")
            if src:
                video_urls.append(src)
            sources = vid.find_elements(By.TAG_NAME, "source")
            for source in sources:
                src = source.get_attribute("src")
                if src:
                    video_urls.append(src)
    except Exception:
        pass
    return {"platform": "YouTube", "title": title, "views": views, "video_urls": video_urls}


def scrape_instagram(driver, url):
    """Scrape metadata Instagram"""
    driver.get(url)
    wait = WebDriverWait(driver, 15)

    try:
        caption = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "h1, div[role='button']"))
        ).text
    except:
        caption = "Unknown"

    # Ambil video url dari halaman (jika ada)
    video_urls = []
    image_urls = []
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for vid in video_elements:
            src = vid.get_attribute("src")
            if src:
                video_urls.append(src)
            sources = vid.find_elements(By.TAG_NAME, "source")
            for source in sources:
                src = source.get_attribute("src")
                if src:
                    video_urls.append(src)
        # Ambil gambar
        image_elements = driver.find_elements(By.TAG_NAME, "img")
        for img in image_elements:
            src = img.get_attribute("src")
            if src and "data:image" not in src:
                image_urls.append(src)
            srcset = img.get_attribute("srcset")
            if srcset:
                first_url = srcset.split(',')[0].strip().split(' ')[0]
                image_urls.append(first_url)
    except Exception:
        pass
    return {"platform": "Instagram", "caption": caption, "video_urls": video_urls, "image_urls": image_urls}


def scrape_tiktok(driver, url):
    """Scrape metadata TikTok"""
    driver.get(url)
    wait = WebDriverWait(driver, 15)

    try:
        caption = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "h1, strong"))
        ).text
    except:
        caption = "Unknown"

    # Ambil video url dari halaman (jika ada)
    video_urls = []
    image_urls = []
    try:
        video_elements = driver.find_elements(By.TAG_NAME, "video")
        for vid in video_elements:
            src = vid.get_attribute("src")
            if src:
                video_urls.append(src)
            sources = vid.find_elements(By.TAG_NAME, "source")
            for source in sources:
                src = source.get_attribute("src")
                if src:
                    video_urls.append(src)
        # Ambil gambar
        image_elements = driver.find_elements(By.TAG_NAME, "img")
        for img in image_elements:
            src = img.get_attribute("src")
            if src and "data:image" not in src:
                image_urls.append(src)
            srcset = img.get_attribute("srcset")
            if srcset:
                first_url = srcset.split(',')[0].strip().split(' ')[0]
                image_urls.append(first_url)
    except Exception:
        pass
    return {"platform": "TikTok", "caption": caption, "video_urls": video_urls, "image_urls": image_urls}


# ============================================================
# Media Downloader
# ============================================================
import os
import yt_dlp
from pytube import YouTube

def download_media(url, output_folder, audio_only=False):
    """Download video/audio pakai yt_dlp atau pytube"""
    os.makedirs(output_folder, exist_ok=True)
    video_file, audio_file = None, None

    if not audio_only:
     # Dalam fungsi download_media, ganti ydl_opts dengan:
        ydl_opts = {
            "outtmpl": f"{output_folder}/%(title)s.%(ext)s",
            "format": "best[height<=720]/best[height<=480]/best",
            "merge_output_format": "mp4",
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # ambil path hasil download (setelah merge)
            video_file = ydl.prepare_filename(info)
            # pastikan extension jadi .mp4
            if not video_file.endswith(".mp4"):
                base, _ = os.path.splitext(video_file)
                video_file = base + ".mp4"

    # khusus YouTube: audio-only pakai pytube
    if "youtube" in url.lower():
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
            audio_file = stream.download(output_path=output_folder)
        except Exception as e:
            print(f"[WARN] Gagal download audio: {e}")

    return video_file, audio_file


# def download_media(url, output_folder, audio_only=False):
#     """Download video/audio pakai yt_dlp atau pytube"""
#     os.makedirs(output_folder, exist_ok=True)
#     video_file, audio_file = None, None

#     # video / audio (yt_dlp)
#     if not audio_only:
#         ydl_opts = {"outtmpl": f"{output_folder}/%(title)s.%(ext)s", 
#                     "format": "bestvideo+bestaudio/best",
#                     "merged_output_format": "mp4"
#                     }
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=True)
#             video_file = ydl.prepare_filename(info)

#     # khusus YouTube: audio-only pakai pytube
#     if "youtube" in url and YouTube:
#         try:
#             yt = YouTube(url)
#             stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
#             audio_file = stream.download(output_path=output_folder)
#         except Exception as e:
#             print(f"[WARN] Gagal download audio: {e}")

#     return video_file, audio_file


# ============================================================
# Universal scrape_to_dataframe
# ============================================================
def scrape_to_dataframe(url: str, save_media: bool = True, output_folder: str = "output_vid") -> pd.DataFrame:
    driver = init_driver()
    if driver is None:
        return pd.DataFrame()

    try:
        domain = urlparse(url).netloc
        data_id = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]

        # tentukan platform
        if "youtube" in domain:
            scraped = scrape_youtube(driver, url)
        elif "instagram" in domain:
            scraped = scrape_instagram(driver, url)
        elif "tiktok" in domain:
            scraped = scrape_tiktok(driver, url)
        else:
            print(f"[WARN] Platform tidak didukung: {domain}")
            return pd.DataFrame()


        # Siapkan field video_urls dan image_urls
        video_urls = scraped.get("video_urls", [])
        image_urls = scraped.get("image_urls", [])



        # Default folder assignment
        image_folder = None
        video_folder = None

        record = {"id": data_id, "url": url, **scraped,
            "video_urls": ", ".join(video_urls) if video_urls else None,
            "image_urls": ", ".join(image_urls) if image_urls else None,
            "video_file": None, "audio_file": None,
            "image_folder": None,
            "video_folder": None,
            "audio_folder": None}

        # download media
        if save_media:
            folder = os.path.join(output_folder, scraped["platform"].lower())
            video_file, audio_file = download_media(url, folder)
            record["video_file"] = video_file
            record["audio_file"] = audio_file

            # download images
            if image_urls:
                image_folder = os.path.join(folder, "images")
                os.makedirs(image_folder, exist_ok=True)
                for idx, img_url in enumerate(image_urls):
                    filename = f"image_{data_id}_{idx+1}.jpg"
                    save_binary(img_url, image_folder, filename)
                record["image_folder"] = image_folder

            # assign video_folder for video platforms (tiktok, instagram)
            if scraped["platform"].lower() in ["tiktok", "instagram"]:
                record["video_folder"] = folder

                # Ekstrak audio dari video jika video_file ada
                if video_file and os.path.isfile(video_file):
                    import subprocess
                    audio_folder = os.path.join(folder, "audio")
                    os.makedirs(audio_folder, exist_ok=True)
                    audio_file_path = os.path.join(audio_folder, f"audio_{data_id}.mp3")
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", video_file, "-vn", "-acodec", "mp3", audio_file_path
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        record["audio_file"] = audio_file_path
                        record["audio_folder"] = audio_folder
                    except Exception as e:
                        print(f"[WARN] Ekstraksi audio gagal: {e}")

        return pd.DataFrame([record])

    except Exception as e:
        print(f"Error scrape_to_dataframe: {e}")
        return pd.DataFrame()

    finally:
        driver.quit()


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.instagram.com/p/C_example/",
        "https://www.tiktok.com/@scout2015/video/6718335390845095173"
    ]

    all_df = pd.DataFrame()
    for u in urls:
        df = scrape_to_dataframe(u, save_media=True, output_folder="downloads")
        all_df = pd.concat([all_df, df], ignore_index=True)

    print(all_df)
    all_df.to_csv("scraped_results.csv", index=False)
    print("[INFO] Data tersimpan ke scraped_results.csv")