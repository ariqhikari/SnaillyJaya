from src.repositories.CleanDataRepository import CleanDataRepository
from src.services.PredictDataServices import PredictDataService
from src.utils.convert import queryResultToDict
from src.services.Service import Service
from src.utils.Scrapping.Scrapping import scrape_to_dataframe
# from src.utils.Scrapping.ScrappingMedsos import scrape_to_dataframe as scrape_medsos
from src.utils.Scrapping.ScrappingMedsos import scrape_to_dataframe as scrape_medsos
from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
from src.utils.Preprocessing.VideoProcessor import caption_videos_in_folder 
from src.utils.errorHandler import errorHandler
from src.utils.Preprocessing.TextProcessor import process_text
import os
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
                    pattern = r"^(instagram\.com|tiktok\.com|youtube\.com|youtu\.be)$"
                    return re.match(pattern, domain) is not None
                except Exception:
                    return False
                
            print(f"Memulai scraping untuk URL: {url}")
            if is_medsos_url(url):
                # scrape_medsos expects a list of URLs and uses 'base_folder' and caption flags
                df_scraped = scrape_medsos([url], base_folder="output_vid", caption_images=True, caption_videos=True)
            else:
                df_scraped = scrape_to_dataframe(url, save_images=True, output_folder="output")

            if df_scraped.empty:
                return self.failedOrSuccessRequest('failed', 404, "Gagal mendapatkan konten dari URL.")

            # Ambil data dari baris pertama DataFrame hasil scraping
            print("DEBUG: sebelum ambil raw_text")
            scraped_result = df_scraped.iloc[0]
            raw_text = scraped_result.get('text') or scraped_result.get('caption') or scraped_result.get('title') or ""
            print("DEBUG: raw_text =", raw_text)
            
            image_links_str = scraped_result.get('image_urls')
            image_folder = scraped_result.get('image_folder')
            video_links_str = scraped_result.get('video_urls')
            video_folder = scraped_result.get('video_folder')

            # Inisialisasi segments_list di sini untuk menghindari error
            segments_list = []
            combined_text = raw_text

            # Image Captioning
            if image_folder and os.path.isdir(image_folder):
                print(f"Memulai image captioning untuk gambar di folder: {image_folder}")
                try:
                    df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
                    if not df_captions.empty:
                        captions_text = ". ".join(df_captions['Caption'].tolist())
                        combined_text = f"{raw_text}. {captions_text}".strip()
                        print(f"✅ Image captioning selesai: {len(df_captions)} gambar diproses")
                    else:
                        print("⚠️ Image captioning tidak menghasilkan data")
                except Exception as e:
                    print(f"❌ Error image captioning: {e}")
                    # Lanjutkan dengan text asli jika image captioning gagal

            # Video Captioning + Speech-to-Text
            elif video_folder and os.path.isdir(video_folder):
                print(f"Memulai video captioning untuk video di folder: {video_folder}")
                try:
                    df_video = caption_videos_in_folder(video_folder)
                    if not df_video.empty:
                        # Process segments dan gabungkan teks
                        combined_texts = []

                        for _, row in df_video.iterrows():
                            visual_caption = row.get('Visual', '')
                            transcript = row.get('Transcript', '')
                            start_time = row.get('Start')
                            end_time = row.get('End')
                            
                            # Skip entries dengan error
                            if visual_caption and any(error in str(visual_caption) for error in ['[ERROR]', '[WARNING]', '[NO CAPTION]']):
                                continue
                            
                            # Buat list segments
                            segment_data = {
                                "start": float(start_time) if start_time is not None else 0.0,
                                "end": float(end_time) if end_time is not None else 0.0,
                                "caption": visual_caption,
                                "transcript": transcript if transcript else ""
                            }
                            segments_list.append(segment_data)
                            
                            # Gabungkan teks untuk combined_text
                            if visual_caption:
                                segment_text = f"{visual_caption}"
                                if transcript:
                                    segment_text += f". {transcript}"
                                combined_texts.append(segment_text)

                        # Update combined_text dengan hasil video processing
                        if combined_texts:
                            video_text = " ".join(combined_texts)
                            combined_text = f"{raw_text}. {video_text}".strip()
                            print(f"✅ Video captioning selesai: {len(segments_list)} segments dibuat")
                        else:
                            print("⚠️ Video captioning tidak menghasilkan caption yang valid")
                    else:
                        print("⚠️ Video captioning tidak menghasilkan data")
                except Exception as e:
                    print(f"❌ Error video captioning: {e}")
                    traceback.print_exc()
                    # Lanjutkan dengan text asli jika video captioning gagal

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
                "segments": segments_list if segments_list else None  # Simpan segments ke DB
            }
            
            print(f"📊 Data yang akan disimpan:")
            print(f"   - URL: {url}")
            print(f"   - Text length: {len(clean_text)}")
            print(f"   - Image folder: {image_folder}")
            print(f"   - Video folder: {video_folder}")
            print(f"   - Segments: {len(segments_list) if segments_list else 0}")
            
            # 5. Panggil Repository untuk menyimpan
            new_record = cleanDataRepository.createNewCleanData(data_to_save)
            record_dict = queryResultToDict([new_record])[0]
            
            data_for_prediction = {
                'text': record_dict['text'],
                'url': record_dict['url'],
                'parent_id': parent_id,
                'child_id': child_id,
            }
            
            prediction_result = predictDataService.createPredictData(data_for_prediction)
            if prediction_result['status'] == 'failed':
                return self.failedOrSuccessRequest('failed', prediction_result['code'], prediction_result['data'])
            
            return self.failedOrSuccessRequest('success', prediction_result['code'], prediction_result['data'])
            
        except ValueError as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 400, f"Terjadi kesalahan internal: {e}")
        except Exception as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, f"Terjadi kesalahan internal: {e}")