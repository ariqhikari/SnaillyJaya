# src/services/ScreenshotDataService.py

# src/services/ScreenshotDataService.py
import os, traceback
import pandas as pd
from src.repositories.ScreenshotDataRepository import ScreenshotDataRepository
from src.services.PredictDataServices import PredictDataService
from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
from src.utils.Preprocessing.TextProcessorBERT import load_bert, caption_to_embedding
from src.utils.Preprocessing.TextProcessor import process_text
from src.utils.convert import queryResultToDict

screenshotRepository = ScreenshotDataRepository()

class ScreenshotDataService:
    def createScreenshotFromFile(self, image_path: str, data: dict, use_bert: bool = True):
        """
        Flow: Screenshot (single image) -> Captioning -> Preprocessing -> (Predict Label via TF-IDF / BERT) -> Save
        """
        # --- Download model from Google Drive only if missing ---
        import os
        model_dir = os.path.join("public", "models", "xlm-roberta-finetuned")
        model_file = os.path.join(model_dir, "config.json")
        drive_folder_url = "https://drive.google.com/drive/folders/136au8fxDDezCbj2b-iDxg24Qdv2Qbd-o?usp=sharing"
        if not os.path.exists(model_file):
            print("Model xlm-roberta-finetuned not found. Mulai proses download dari Google Drive...")
            print("Mohon tunggu, proses download model dari Google Drive sedang berlangsung...")
            try:
                import gdown
                gdown.download_folder(url=drive_folder_url, output=model_dir, quiet=False, use_cookies=False)
                print("Download model dari Google Drive selesai.")
            except Exception as e:
                print(f"Gagal download model dari Google Drive: {e}")
                return self.failedOrSuccessRequest('failed', 500, f"Model download failed: {e}")
        else:
            print("Model xlm-roberta-finetuned sudah ada. Lewati proses download.")

        try:
            if not os.path.isfile(image_path):
                return self.failedOrSuccessRequest('failed', 400, "File gambar tidak ditemukan.")

            # 1. Captioning untuk satu gambar
            print("=== CAPTIONING PROCESS (single image) ===")
            from src.utils.Preprocessing.ImageProcessor import caption_image
            caption = caption_image(image_path)
            if not caption:
                return self.failedOrSuccessRequest('failed', 404, "Caption gagal dibuat.")

            captions_text = caption

            # 2. Preprocessing
            from src.utils.Preprocessing.TextProcessor import process_text
            clean_text, final_tokens = process_text(captions_text)
            if not clean_text:
                return self.failedOrSuccessRequest('failed', 500, "Teks kosong setelah preprocessing.")

            # 3. Predict Label dengan Fallback Mechanism
            label = None
            bert_failed = False
            embedding = None


            if use_bert:
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    inputs = tokenizer(captions_text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)
                        confidence, predicted_label_idx = torch.max(probs, dim=1)
                        confidence = confidence.item()
                        predicted_label = predicted_label_idx.item()
                    if predicted_label == 1:
                        label = "berbahaya"
                    else:
                        label = "aman"
                    print(f"BERT Prediction - Confidence: {confidence:.4f}, Label: {label}")
                except Exception as e:
                    print(f"Error prediksi BERT: {e}")
                    label = "unknown"
                # Embedding tetap bisa diambil jika perlu
                try:
                    from src.utils.Preprocessing.TextProcessorBERT import load_bert, caption_to_embedding
                    tokenizer_bert, model_bert = load_bert()
                    embedding = caption_to_embedding(captions_text, tokenizer_bert, model_bert)
                    if embedding is not None:
                        embedding = embedding.tolist()
                except Exception as e:
                    print(f"Error creating BERT embedding: {e}")
                    embedding = None
            else:
                try:
                    label = PredictDataService().predictLabel({"text": clean_text})
                    print(f"TF-IDF Prediction - Label: {label}")
                except Exception as e:
                    print(f"Error TF-IDF prediction: {e}")
                    label = "unknown"

            if label is None or label == "":
                label = "unknown"
                print("Label masih kosong, set ke 'unknown'")

            # 5. Save to DB Screenshot
            data_to_save = {
                "text": clean_text,
                "raw_text": captions_text,
                "stopword_removed_tokens": final_tokens,
                "file_gambar": image_path,
                "folder_gambar": os.path.dirname(image_path),
                "label": label,
                "embedding": embedding
            }
            new_record = screenshotRepository.createNewScreenshotData(data_to_save)
            record_dict = queryResultToDict([new_record])[0]
            print("Record saved dengan label:", label)
            return self.failedOrSuccessRequest('success', 200, record_dict)

        except Exception as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))
    def failedOrSuccessRequest(self, status, code, data):
        return {
            "status": status,
            "code": code,
            "data": data,
        }

    def createScreenshotData(self, image_folder: str, data: dict, use_bert: bool = True):
        """
        Flow: Screenshot -> Captioning -> Preprocessing -> (Predict Label via TF-IDF / BERT) -> Save
        Dengan fallback ke TF-IDF jika BERT gagal
        """
        # --- Download model from Google Drive only if missing ---
        import os
        import gdown
        model_dir = os.path.join("public", "models", "xlm-roberta-finetuned")
        model_file = os.path.join(model_dir, "config.json")
        drive_folder_url = "https://drive.google.com/drive/folders/136au8fxDDezCbj2b-iDxg24Qdv2Qbd-o?usp=sharing"
        if not os.path.exists(model_file):
            print("Model xlm-roberta-finetuned not found. Mulai proses download dari Google Drive...")
            print("Mohon tunggu, proses download model dari Google Drive sedang berlangsung...")
            try:
                import gdown
                gdown.download_folder(url=drive_folder_url, output=model_dir, quiet=False, use_cookies=False)
                print("Download model dari Google Drive selesai.")
            except Exception as e:
                print(f"Gagal download model dari Google Drive: {e}")
                return self.failedOrSuccessRequest('failed', 500, f"Model download failed: {e}")
        else:
            print("Model xlm-roberta-finetuned sudah ada. Lewati proses download.")

        try:
            if not os.path.isdir(image_folder):
                return self.failedOrSuccessRequest('failed', 400, "Folder gambar tidak ditemukan.")

            # 1. Captioning
            print("=== CAPTIONING PROCESS ===")
            df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
            if df_captions.empty:
                return self.failedOrSuccessRequest('failed', 404, "Tidak ada caption yang berhasil dibuat.")
            
            # Gabungkan semua caption jadi satu teks besar
            captions_text = ". ".join(df_captions['Caption'].tolist())

            # 2. Preprocessing (untuk TF-IDF/analisis tradisional)
            clean_text, final_tokens = process_text(captions_text)
            if not clean_text:
                return self.failedOrSuccessRequest('failed', 500, "Teks kosong setelah preprocessing.")

            # 3. Predict Label dengan Fallback Mechanism
            label = None
            bert_failed = False
            embedding = None
            bert_confidence = None
            bert_label = None
            used_model = ""

            if use_bert:
                # Thresholding + confidence check
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch
                    model_dir = os.path.join("public", "models", "xlm-roberta-finetuned")
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                    inputs = tokenizer(captions_text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)
                        confidence, predicted_label_idx = torch.max(probs, dim=1)
                        confidence = confidence.item()
                        predicted_label = predicted_label_idx.item()
                    threshold = 0.7  # Atur sesuai kebutuhan
                    
                    if predicted_label == 1:
                        label = "berbahaya"
                    else:
                        label = "aman"
                    
                    # Check jika confidence rendah
                    if confidence < threshold:
                        print(f"Prediksi BERT confidence rendah: {confidence:.2f}, label: {label}. Fallback ke TF-IDF.")
                        bert_failed = True
                    else:
                        print(f"BERT Prediction - Confidence: {confidence:.4f}, Label: {label}")
                        
                except Exception as e:
                    print(f"Error prediksi BERT: {e}")
                    print("Fallback ke TF-IDF prediction...")
                    bert_failed = True
                
                # Embedding tetap bisa diambil jika perlu
                try:
                    tokenizer_bert, model_bert = load_bert()
                    embedding = caption_to_embedding(captions_text, tokenizer_bert, model_bert)
                    if embedding is not None:
                        embedding = embedding.tolist()
                except Exception as e:
                    print(f"Error creating BERT embedding: {e}")
                    embedding = None

            # Fallback ke TF-IDF jika BERT gagal atau confidence rendah
            if (use_bert and bert_failed) or not use_bert:
                try:
                    label = PredictDataService().predictLabel({"text": clean_text})
                    print(f"TF-IDF Prediction - Label: {label}")
                except Exception as e:
                    print(f"Error TF-IDF prediction: {e}")
                    label = "unknown"
            
            # Jika masih None, set default
            if label is None or label == "":
                label = "unknown"
                print("Label masih kosong, set ke 'unknown'")

            # 5. Save to DB Screenshot
            data_to_save = {
                "text": clean_text,                   # teks sudah diproses
                "raw_text": captions_text,            # teks asli
                "stopword_removed_tokens": final_tokens,
                "folder_gambar": image_folder,
                "label": label,
                "embedding": embedding                # optional (bisa None)
            }
            new_record = screenshotRepository.createNewScreenshotData(data_to_save)
            record_dict = queryResultToDict([new_record])[0]
            print("Record saved dengan label:", label)

            return self.failedOrSuccessRequest('success', 200, record_dict)

        except Exception as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))
