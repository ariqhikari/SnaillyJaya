
from src.repositories.CleanDataRepository import CleanDataRepository
from src.services.PredictDataServices import PredictDataService
from src.utils.convert import queryResultToDict
from src.services.Service import Service
from src.utils.Scrapping.Scrapping import scrape_to_dataframe
from src.utils.Scrapping.ScrappingMedsos import scrape_to_dataframe as scrape_medsos
from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
# from src.utils.Preprocessing.GITImageProcessor import caption_images_in_folder
from src.utils.Preprocessing.VideoProcessor import caption_videos_in_folder 
from src.utils.errorHandler import errorHandler
from src.utils.Preprocessing.TextProcessor import process_text
import os # Pastikan os diimpor
import json
import re
from urllib.parse import urlparse
import traceback

cleanDataRepository = CleanDataRepository()
predictDataService = PredictDataService()

class CleanDataService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }

    def getAllCleanData(self):
        try:
            data = cleanDataRepository.getAllCleanData()
            return cleanDataRepository.failedOrSuccessRequest('success', 200, queryResultToDict(data))
        except Exception as e:
            return cleanDataRepository.failedOrSuccessRequest('failed', 500, str(e))

    def createCleanData(self, data):
        """
        Menerima data dari Controller (misal: url, parent_id, child_id),
        melakukan scraping, preprocessing, lalu menyimpan ke database.
        
        Args:
            data (dict): Dictionary berisi 'url', 'parent_id', 'child_id'.
        """
        try:
            url = data['url']
            parent_id = data['parent_id']
            child_id = data['child_id']
            # 1. Cek apakah URL sudah ada di database
            existing = cleanDataRepository.getCleanDataByUrl(url)
            if existing:
                existing_data = queryResultToDict([existing])[0]
                print(f"URL sudah ada: {url}. Melewatkan scraping, langsung ke prediksi.")
                data_for_prediction = {
                    'text': existing_data['text'],
                    'url': existing_data['url'],
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'token': data['token']
                }
                prediction_result = predictDataService.createPredictData(data_for_prediction)
                if prediction_result['status'] == 'failed':
                    return self.failedOrSuccessRequest('failed', prediction_result['code'], prediction_result['data'])
                return self.failedOrSuccessRequest('success', prediction_result['code'], prediction_result['data'])
            # 2. Lakukan Scraping
            def is_medsos_url(url: str) -> bool:
                """
                Deteksi apakah URL adalah Instagram, TikTok, atau YouTube.
                Menggunakan urlparse + regex biar aman.
                """
                try:
                    domain = urlparse(url).netloc.lower()

                    # Hilangkan "www." biar lebih konsisten
                    domain = domain.replace("www.", "")

                    # Regex cek domain utama
                    pattern = r"^(instagram\.com|tiktok\.com|youtube\.com)$"
                    return re.match(pattern, domain) is not None
                except Exception:
                    return False
                
            print(f"Memulai scraping untuk URL: {url}")
            if is_medsos_url(url):
                df_scraped = scrape_medsos(url, save_media=True, output_folder="output_vid")
            else:
                df_scraped = scrape_to_dataframe(url, save_images=True, output_folder="output")

            if df_scraped.empty:
                return self.failedOrSuccessRequest('failed', 404, "Gagal mendapatkan konten dari URL.")

            """
            UNTUK GAMBAR
            PR!!: Tambahkan juga kalau bentuknya video harus gimana
            """
            # Ambil data dari baris pertama DataFrame hasil scraping
            # scraped_result = df_scraped.iloc[0]
            # raw_text = scraped_result['text']
            # image_links_str = scraped_result['image_urls']
            # image_folder = scraped_result['image_folder']
            # video_links_str = scraped_result.get('video_urls')
            # video_folder = scraped_result['video_folder']
            print("DEBUG: sebelum ambil raw_text")
            scraped_result = df_scraped.iloc[0]
            raw_text = scraped_result.get('text') or scraped_result.get('caption') or scraped_result.get('title') or ""
            print("DEBUG: raw_text =", raw_text)
            image_links_str = scraped_result.get('image_urls')
            image_folder = scraped_result.get('image_folder')
            video_links_str = scraped_result.get('video_urls')
            video_folder = scraped_result.get('video_folder')

            # TAMBAHKAN UNTUK MELAKUKAN VIDEO CAPTIONING 
            """
            PR!!: buat kondisi, apakah harus melakukan image captioning atau video captioning
            IMAGE CAPTIONING
            """
            # 3. Lakukan Image Captioning
            # combined_text = raw_text
            # if image_folder:
            #     print(f"Memulai image captioning untuk gambar di folder: {image_folder}")
            #     captions_text = ""
            #     # Pastikan folder gambar benar-benar ada dan tidak kosong
            #     if image_folder and os.path.isdir(image_folder):
            #         print(f"Memulai image captioning untuk gambar di folder: {image_folder}")
            #         df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
                    
            #         if not df_captions.empty:
            #             # Ambil semua caption dan gabungkan menjadi satu string
            #             captions_list = df_captions['Caption'].tolist()
            #             captions_text = '. '.join(captions_list)
            #             print(f"Caption yang dihasilkan: {captions_text[:150]}...")
            #         else:
            #             print("Tidak ada caption yang berhasil dibuat.")
            #     else:
            #         print("Tidak ada folder gambar untuk diproses captioning.")


            #     """
            #     PR!!: Untuk combined_text pada video, artinya adalah campuran antara hasil video captioning dengan speech to text
            #     """
            #     # 4. Gabungkan Teks Scraped dengan Teks Caption
            #     # Tambahkan spasi dan titik untuk pemisah yang jelas
            #     combined_text = f"{raw_text}. {captions_text}".strip()

            #CAPTIONING

            combined_text = raw_text
            # Kalau gambar → image captioning
            if image_folder:
                print(f"Memulai image captioning untuk gambar di folder: {image_folder}")
                if os.path.isdir(image_folder):
                    df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
                    if not df_captions.empty:
                        captions_text = ". ".join(df_captions['Caption'].tolist())
                        combined_text = f"{raw_text}. {captions_text}".strip()
            # Kalau video → video captioning + speech-to-text
            elif video_folder:
                print(f"Memulai video captioning untuk video di folder: {video_folder}")
                if os.path.isdir(video_folder):
                    df_video = caption_videos_in_folder(video_folder, delete_corrupted=False)
                    if not df_video.empty:
                        # Campur caption + transcript
                        combined_texts = []
                        segments_list = []

                        for _, row in df_video.iterrows():
                            # combined_texts.append(f"{row['Caption']}. {row['Transcript']}")
                            caption = row.get('Caption', '')
                            transcript = row.get('Transcript', '')
                            start_time = row.get('start_time')
                            end_time = row.get('end_time')

                            #membuat list segments
                            segments_list.append({
                                "start": float(start_time) if start_time else None,
                                "end": float(end_time) if end_time else None,
                                "caption": caption,
                                "transcript": transcript
                            })
                            combined_texts.append(f"{caption}. {transcript}")

                        video_text = " ".join(combined_texts)
                        combined_text = f"{raw_text}. {video_text}".strip()

            # 3. Lakukan Preprocessing Teks
            print("Memulai preprocessing teks...")
            clean_text, final_tokens = process_text(combined_text)
            if not clean_text:
                return self.failedOrSuccessRequest('failed', 500, "Teks kosong setelah preprocessing.")
            print("Preprocessing teks selesai.")

            # 4. Siapkan data untuk disimpan
            data_to_save = {
                "url": url,
                "text": clean_text, 
                "raw_text": combined_text,
                "stopword_removed_tokens": final_tokens,
                "link_gambar": image_links_str,
                "folder_gambar": image_folder,
                "link_video": video_links_str,      
                "folder_video": video_folder,
                "segments": segments_list if video_folder else None #akan disimpan di DB        
            }
            # print(f"Data yang akan disimpan: {data_to_save}")
            
            # 5. Panggil Repository untuk menyimpan
            new_record = cleanDataRepository.createNewCleanData(data_to_save)
            record_dict = queryResultToDict([new_record])[0]
            data_for_prediction = {
                    'text': record_dict['text'],
                    'url': record_dict['url'],
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'token': data['token']
                }
            prediction_result= predictDataService.createPredictData(data_for_prediction)
            if prediction_result['status'] == 'failed':
                return self.failedOrSuccessRequest('failed', prediction_result['code'], prediction_result['data'])
            return self.failedOrSuccessRequest('success', prediction_result['code'], prediction_result['data'])
        except ValueError as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 400, f"Terjadi kesalahan internal: {e}") # Bad request (misal: data tidak lengkap)
        except Exception as e:
            # Tangani error umum lainnya
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, f"Terjadi kesalahan internal: {e}")
