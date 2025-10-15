
import os
import re
import pandas as pd
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import hashlib
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

#memuat JINA AI
def jina_ai(url):
    # Kirim permintaan GET ke URL
    response = requests.get(f"https://r.jina.ai/{url}")
    response.raise_for_status()  # Periksa jika permintaan berhasil

    # Simpan hasil ke dalam list
    full_text = str(response.text)
    return full_text

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

# PERBAIKAN 2: Logika scraping yang lebih tangguh
def scrape_content(driver, url):
    """
    Melakukan scraping pada URL untuk mendapatkan teks dan URL gambar
    dengan penanganan race condition yang lebih baik.
    """
    try:
        print(f"Mulai scraping: {url}")
        driver.get(url)

        wait = WebDriverWait(driver, 20)
        
        # 1. Tentukan locator yang lebih spesifik jika memungkinkan. Untuk Kompas, ini sangat efektif.
        #    Kita tetap sediakan fallback untuk situs lain.
        specific_locator = "//div[contains(@class, 'read__content')]"
        general_locators = "//main | //article | //div[@role='main']"
        
        main_element = None
        try:
            # Coba cari locator spesifik dulu
            main_element = wait.until(EC.presence_of_element_located((By.XPATH, specific_locator)))
            print("Elemen konten spesifik ('read__content') berhasil ditemukan.")
        except TimeoutException:
            # Jika tidak ketemu, baru cari locator umum
            print("Elemen spesifik tidak ditemukan, mencoba locator umum...")
            try:
                main_element = wait.until(EC.presence_of_element_located((By.XPATH, general_locators)))
                print("Elemen konten umum berhasil ditemukan.")
            except TimeoutException:
                print("Semua elemen konten tidak ditemukan, menggunakan 'body' sebagai fallback.")
                main_element = driver.find_element(By.TAG_NAME, "body")

        # 2. Scroll halaman untuk memicu semua lazy-loading
        print("Melakukan scroll halaman...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1) # Beri waktu render setelah scroll

        # 3. SOLUSI UTAMA: Tunggu hingga setidaknya satu gambar muncul DI DALAM kontainer
        try:
            print("Menunggu gambar di dalam kontainer untuk dimuat...")
            # Locator .//img mencari img HANYA di dalam main_element
            wait.until(EC.presence_of_all_elements_located((By.XPATH, ".//img")))
            print("Gambar terdeteksi di dalam kontainer.")
        except TimeoutException:
            print("PERINGATAN: Tidak ada gambar yang ditemukan di dalam kontainer setelah menunggu.")
            # Proses tetap lanjut untuk mengambil teks
        
        # 4. Hapus elemen yang tidak relevan
        driver.execute_script("""
            const container = arguments[0];
            const selectors = ['nav', 'header', 'footer', 'aside', 'script', 'style', '.iklan', '.ads'];
            selectors.forEach(sel => container.querySelectorAll(sel).forEach(el => el.remove()));
        """, main_element)

        # 5. Ambil teks dan URL gambar
        text = main_element.text.strip()
        image_elements = main_element.find_elements(By.TAG_NAME, "img")
        images = set()
        exclude_pattern = re.compile(r'logo|icon|svg|avatar|spinner|loading|pixel|gif', re.I)

        for img in image_elements:
            src = img.get_attribute("src") or img.get_attribute("data-src")
            if src and "data:image" not in src:
                full_url = urljoin(url, src)
                if not exclude_pattern.search(full_url):
                    images.add(full_url)
            
            srcset = img.get_attribute("srcset")
            if srcset:
                first_url = urljoin(url, srcset.split(',')[0].strip().split(' ')[0])
                if not exclude_pattern.search(first_url):
                    images.add(first_url)
        
        images_list = list(images)
        print(f"Scraping selesai. Ditemukan {len(images_list)} gambar.")
        return {'text': text, 'images': images_list}

    except Exception as e:
        print(f"Error fatal saat melakukan scraping pada {url}: {e}")
        # Tambahkan traceback untuk debugging lebih dalam jika perlu
        import traceback
        traceback.print_exc()
        return None

def save_scraped_images(image_urls, base_folder, url, data_id):
    """Menyimpan gambar dari URL ke folder lokal."""
    # Membuat nama folder yang unik dan aman
    parsed = urlparse(url)
    domain = parsed.netloc.replace(':', '_').replace('.', '_')
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:8]
    folder_name = os.path.join(base_folder, f"images_{data_id}_{domain}_{url_hash}")
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    saved_paths = []
    print(f"Menyimpan gambar ke folder: {folder_name}")
    for i, img_url in enumerate(image_urls, start=1):
        try:
            resp = requests.get(img_url, stream=True, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            
            # Ekstensi file
            ext = os.path.splitext(img_url.split('?')[0])[1] or '.jpg'
            if len(ext) > 5 or not ext.startswith('.'):
                ext = '.jpg'
            
            filename = f"image_{data_id}_{i}{ext}"
            path = os.path.join(folder_name, filename)
            
            with open(path, 'wb') as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            saved_paths.append(path)
        except Exception as ex:
            print(f"Gagal menyimpan {img_url}: {ex}")
            
    return folder_name, saved_paths


# --- Main Function ---

def scrape_to_dataframe(url: str, save_images: bool = False, output_folder: str = 'output') -> pd.DataFrame:
    """
    Scraping konten dari URL menggunakan Jina AI terlebih dahulu.
    Jika gagal, fallback ke Selenium scraping.
    """
    data_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
    record = {
        'id': data_id,
        'url': url,
        'text': '',
        'image_urls': '',
        'image_folder': None,
        'saved_image_paths': None
    }

    # --- Tahap 1: Coba Jina AI ---
    try:
        print(f"Mencoba scraping via Jina AI: {url}")
        jina_text = jina_ai(url)
        if jina_text and len(jina_text.strip()) > 50:  # validasi hasil
            print("✅ Berhasil mengambil teks menggunakan Jina AI.")
            record['text'] = jina_text
            df = pd.DataFrame([record])
            return df
        else:
            print("⚠️ Hasil Jina AI kosong atau terlalu pendek, lanjut ke Selenium.")
    except Exception as e:
        print(f"⚠️ Jina AI gagal: {e}, lanjut ke Selenium.")

    # --- Tahap 2: Fallback ke Selenium ---
    driver = init_driver()
    if driver is None:
        print("❌ Gagal menginisialisasi WebDriver. Tidak dapat melakukan scraping Selenium.")
        return pd.DataFrame()

    try:
        scraped_data = scrape_content(driver, url)
        if not scraped_data:
            print("Tidak ada data yang bisa di-scrape menggunakan Selenium.")
            return pd.DataFrame()

        record['text'] = scraped_data['text']
        record['image_urls'] = ', '.join(scraped_data['images'])

        if save_images and scraped_data['images']:
            folder_path, saved_paths = save_scraped_images(
                image_urls=scraped_data['images'],
                base_folder=output_folder,
                url=url,
                data_id=data_id
            )
            record['image_folder'] = folder_path
            record['saved_image_paths'] = ', '.join(saved_paths)
            print(f"Total {len(saved_paths)} gambar berhasil disimpan.")

        df = pd.DataFrame([record])
        return df

    except Exception as e:
        print(f"Terjadi error saat scraping Selenium: {e}")
        return pd.DataFrame()

    finally:
        print("Menutup WebDriver.")
        driver.quit()

def scrape_to_dataframe(url: str, save_images: bool = False, output_folder: str = 'output') -> pd.DataFrame:
    """
    Fungsi utama untuk melakukan scraping pada satu URL dan mengembalikan hasilnya
    sebagai Pandas DataFrame.

    Args:
        url (str): URL dari halaman web yang ingin di-scrape.
        save_images (bool, optional): Jika True, akan mengunduh gambar yang ditemukan.
                                      Defaultnya adalah False.
        output_folder (str, optional): Nama folder untuk menyimpan gambar jika save_images=True.
                                       Defaultnya adalah 'output'.

    Returns:
        pd.DataFrame: Sebuah DataFrame yang berisi data hasil scraping.
                      Mengembalikan DataFrame kosong jika terjadi error.
    """
    driver = init_driver()
    if driver is None:
        return pd.DataFrame()

    try:
        # 1. Scrape konten dari halaman web
        scraped_data = scrape_content(driver, url)

        if not scraped_data:
            print("Tidak ada data yang bisa di-scrape.")
            return pd.DataFrame()

        # Siapkan data untuk DataFrame
        data_id = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
        record = {
            'id': data_id,
            'url': url,
            'text': scraped_data['text'],
            'image_urls': ', '.join(scraped_data['images']),
            'image_folder': None, # Akan diisi jika gambar disimpan
            'saved_image_paths': None # Akan diisi jika gambar disimpan
        }
        
        # 2. Simpan gambar jika diminta
        if save_images and scraped_data['images']:
            folder_path, saved_paths = save_scraped_images(
                image_urls=scraped_data['images'],
                base_folder=output_folder,
                url=url,
                data_id=data_id
            )
            record['image_folder'] = folder_path
            record['saved_image_paths'] = ', '.join(saved_paths)
            print(f"Total {len(saved_paths)} gambar berhasil disimpan.")

        # 3. Buat DataFrame
        df = pd.DataFrame([record])
        return df

    except Exception as e:
        print(f"Terjadi error tak terduga dalam proses: {e}")
        return pd.DataFrame()

    finally:
        # 4. Pastikan driver selalu ditutup
        if driver:
            print("Menutup WebDriver.")
            driver.quit()
