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
        try:
            if not os.path.isdir(image_folder):
                return self.failedOrSuccessRequest('failed', 400, "Folder gambar tidak ditemukan.")

            # 1. Captioning
            print("=== CAPTIONING PROCESS ===")
            df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
            if df_captions.empty:
                return self.failedOrSuccessRequest('failed', 404, "Tidak ada caption yang berhasil dibuat.")
            
            print(f"Generated captions: {df_captions['Caption'].tolist()}")
            
            # Gabungkan semua caption jadi satu teks besar
            captions_text = ". ".join(df_captions['Caption'].tolist())
            print(f"Combined captions text: {captions_text}")

            # 2. Preprocessing (untuk TF-IDF/analisis tradisional)
            print("=== TEXT PREPROCESSING ===")
            clean_text, final_tokens = process_text(captions_text)
            print(f"Original text: {captions_text}")
            print(f"Clean text: {clean_text}")
            print(f"Final tokens: {final_tokens}")
            
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
                    inputs = tokenizer(captions_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=1)
                        confidence, predicted_label_idx = torch.max(probs, dim=1)
                        confidence = confidence.item()
                        predicted_label = predicted_label_idx.item()
                    
                    bert_confidence = confidence
                    bert_label = "berbahaya" if predicted_label == 1 else "aman"
                    threshold = 0.5
                    
                    # DEBUG: Print confidence details
                    print(f"=== BERT PREDICTION DEBUG ===")
                    print(f"Input text length: {len(captions_text)}")
                    print(f"BERT Confidence: {confidence:.4f}")
                    print(f"BERT Raw Prediction: {predicted_label} -> {bert_label}")
                    print(f"Probability distribution: {probs.tolist()}")
                    print(f"Threshold: {threshold}")
                    
                    if confidence < threshold:
                        print(f"BERT confidence rendah ({confidence:.4f} < {threshold}), fallback ke TF-IDF")
                        bert_failed = True
                    else:
                        label = bert_label
                        used_model = "BERT"
                        print(f"Using BERT prediction: {label} with confidence {confidence:.4f}")
                        
                    # Embedding tetap bisa diambil jika perlu
                    try:
                        tokenizer_bert, model_bert = load_bert()
                        embedding = caption_to_embedding(captions_text, tokenizer_bert, model_bert)
                        if embedding is not None:
                            embedding = embedding.tolist()
                            print(f"BERT embedding created, shape: {len(embedding)}")
                    except Exception as e:
                        print(f"Error creating BERT embedding: {e}")
                        embedding = None
                        
                except Exception as e:
                    print(f"Error prediksi BERT: {e}")
                    print("Fallback ke TF-IDF prediction...")
                    bert_failed = True

            # Fallback ke TF-IDF jika BERT gagal atau confidence rendah
            if (use_bert and bert_failed) or not use_bert:
                try:
                    print("=== TF-IDF PREDICTION ===")
                    tfidf_prediction = PredictDataService().predictLabel({"text": clean_text})
                    label = tfidf_prediction
                    used_model = "TF-IDF"
                    print(f"TF-IDF Prediction: {label}")
                    print(f"Input to TF-IDF: {clean_text}")
                except Exception as e:
                    print(f"Error TF-IDF prediction: {e}")
                    label = "unknown"
            
            # Jika masih None, set default
            if label is None or label == "":
                label = "unknown"
                used_model = "DEFAULT"
                print("Label masih kosong, set ke 'unknown'")

            # DEBUG: Final decision
            print(f"=== FINAL DECISION ===")
            print(f"Used model: {used_model}")
            print(f"Final label: {label}")
            print(f"BERT confidence: {bert_confidence}")
            print(f"BERT label: {bert_label}")

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
            print("✅ Record saved dengan label:", label)

            return self.failedOrSuccessRequest('success', 200, record_dict)

        except Exception as e:
            print("=== ERROR ===")
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))

# import os, traceback
# import pandas as pd
# from src.repositories.ScreenshotDataRepository import ScreenshotDataRepository
# from src.services.PredictDataServices import PredictDataService
# from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
# from src.utils.Preprocessing.TextProcessorBERT import load_bert, caption_to_embedding
# from src.utils.Preprocessing.TextProcessor import process_text
# from src.utils.convert import queryResultToDict

# screenshotRepository = ScreenshotDataRepository()

# class ScreenshotDataService:
#     def failedOrSuccessRequest(self, status, code, data):
#         return {
#             "status": status,
#             "code": code,
#             "data": data,
#         }

#     def createScreenshotData(self, image_folder: str, data: dict, use_bert: bool = True):
#         """
#         Flow: Screenshot -> Captioning -> Preprocessing -> (Predict Label via TF-IDF / BERT) -> Save
#         CATATAN:
#         1. Dahulukan predict menggunakan BERT jika use_bert=True, jika tidak gunakan TF-IDF.
#         """
#         try:
#             if not os.path.isdir(image_folder):
#                 return self.failedOrSuccessRequest('failed', 400, "Folder gambar tidak ditemukan.")

#             # 1. Captioning
#             df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
#             if df_captions.empty:
#                 return self.failedOrSuccessRequest('failed', 404, "Tidak ada caption yang berhasil dibuat.")
            
#             # Gabungkan semua caption jadi satu teks besar
#             captions_text = ". ".join(df_captions['Caption'].tolist())

#             # 2. Preprocessing (untuk TF-IDF/analisis tradisional)
#             clean_text, final_tokens = process_text(captions_text)
#             if not clean_text:
#                 return self.failedOrSuccessRequest('failed', 500, "Teks kosong setelah preprocessing.")

#             # 3. Embedding (opsional untuk BERT)
#             embedding = None
#             label = None
#             if use_bert:
#                 # Thresholding + confidence check
#                 try:
#                     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#                     import torch
#                     model_dir = os.path.join("public", "models", "xlm-roberta-finetuned")
#                     tokenizer = AutoTokenizer.from_pretrained(model_dir)
#                     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
#                     inputs = tokenizer(captions_text, return_tensors="pt", truncation=True, padding=True)
#                     with torch.no_grad():
#                         outputs = model(**inputs)
#                         logits = outputs.logits
#                         probs = torch.softmax(logits, dim=1)
#                         confidence, predicted_label_idx = torch.max(probs, dim=1)
#                         confidence = confidence.item()
#                         predicted_label = predicted_label_idx.item()
#                     threshold = 0.7  # Atur sesuai kebutuhan
#                     if predicted_label == 1:
#                         label = "berbahaya"
#                     else:
#                         label = "aman"
#                     if confidence < threshold:
#                         print(f"Prediksi BERT confidence rendah: {confidence:.2f}, label: {label}")
#                 except Exception as e:
#                     print("Error prediksi BERT dengan confidence:", e)
#                     label = ""
#                 # Embedding tetap bisa diambil jika perlu
#                 tokenizer_bert, model_bert = load_bert()
#                 embedding = caption_to_embedding(captions_text, tokenizer_bert, model_bert)
#                 if embedding is not None:
#                     embedding = embedding.tolist()
#             else:
#                 # Prediksi label pakai clean_text (default TF-IDF pipeline)
#                 label = PredictDataService().predictLabel({"text": clean_text})

#             # 5. Save to DB Screenshot
#             data_to_save = {
#                 "text": clean_text,                   # teks sudah diproses
#                 "raw_text": captions_text,            # teks asli
#                 "stopword_removed_tokens": final_tokens,
#                 "folder_gambar": image_folder,
#                 "label": label,
#                 "embedding": embedding                # optional (bisa None)
#             }
#             new_record = screenshotRepository.createNewScreenshotData(data_to_save)
#             record_dict = queryResultToDict([new_record])[0]
#             print("Record saved:", record_dict)

#             return self.failedOrSuccessRequest('success', 200, record_dict)

#         except Exception as e:
#             traceback.print_exc()
#             return self.failedOrSuccessRequest('failed', 500, str(e))


# # src/services/ScreenshotDataService.py
# from src.repositories.ScreenshotDataRepository import ScreenshotDataRepository
# from src.services.PredictDataServices import PredictDataService
# from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
# from src.utils.Preprocessing.TextProcessor import process_text
# from src.utils.convert import queryResultToDict
# import os, traceback

# screenshotRepository = ScreenshotDataRepository()

# class ScreenshotDataService:
#     def failedOrSuccessRequest(self, status, code, data):
#         return {
#             "status": status,
#             "code": code,
#             "data": data,
#         }

#     def createScreenshotData(self, image_folder: str, data: dict):
#         """
#         Flow: Screenshot -> Captioning -> Preprocessing -> Predict Label -> Save
#         """
#         try:
#             if not os.path.isdir(image_folder):
#                 return self.failedOrSuccessRequest('failed', 400, "Folder gambar tidak ditemukan.")

#             # 1. Captioning
#             df_captions = caption_images_in_folder(image_folder, delete_corrupted=False)
#             if df_captions.empty:
#                 return self.failedOrSuccessRequest('failed', 404, "Tidak ada caption yang berhasil dibuat.")
#             captions_text = ". ".join(df_captions['Caption'].tolist())

#             # 2. Preprocessing
#             clean_text, final_tokens = process_text(captions_text)
#             if not clean_text:
#                 return self.failedOrSuccessRequest('failed', 500, "Teks kosong setelah preprocessing.")


#             # 3. Label Prediksi (pastikan string, max 255 karakter)
#             label = PredictDataService().predictLabel({"text": clean_text})

#             # 4. Save to DB Screenshot
#             data_to_save = {
#                 "text": clean_text,
#                 "raw_text": captions_text,
#                 "stopword_removed_tokens": final_tokens,
#                 "folder_gambar": image_folder,
#                 "label": label
#             }
#             new_record = screenshotRepository.createNewScreenshotData(data_to_save)
#             record_dict = queryResultToDict([new_record])[0]
#             print("Record saved:", record_dict)

#             return self.failedOrSuccessRequest('success', 200, record_dict)

#         except Exception as e:
#             traceback.print_exc()
#             return self.failedOrSuccessRequest('failed', 500, str(e))
