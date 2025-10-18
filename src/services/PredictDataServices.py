from src.repositories.PredictDataRepository import PredictDataRepository
from src.repositories.UrlClassificationRepository import UrlClassificationRepository
from src.repositories.CleanDataRepository import CleanDataRepository
from src.repositories.LogActivityRepository import LogActivityRepository
from src.config.config import API_SNAILLY
from src.utils.convert import queryResultToDict
from src.services.Service import Service
from src.utils.errorHandler import errorHandler
import os # Pastikan os diimpor
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')
import ast
import requests  # Masih dibutuhkan untuk sendNotification
from urllib.parse import urlparse

predictDataRepository = PredictDataRepository()
urlClassificationRepository = UrlClassificationRepository()
cleanDataRepository = CleanDataRepository()
logActivityRepository = LogActivityRepository()

class PredictDataService(Service):
    # =============================
    # Utility: Normalisasi teks
    # =============================
    def normalize_text(self, text: str) -> str:
        import re
        if not isinstance(text, str):
            return ""
        # hapus repetisi kata
        text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        } 
    
    def __init__(self):
        self.svm_model, self.tfidf_vectorizer = self._load_model()

        # untuk BERT
        self.xlm_tokenizer = None
        self.xlm_model = None
        self.device = None

    def sendLog(self, childId, parentId, url):
        """
        Insert log_activity langsung ke database.
        Return log_id jika berhasil, None jika gagal.
        """
        try:
            # Validasi input
            if not childId or not url:
                print("❌ Error: childId dan url wajib diisi")
                return None
            
            log_data = {
                "childId": str(childId),
                "url": str(url),
                "parentId": str(parentId) if parentId else None,
            }
            
            print(f"📝 Attempting to create log with data: {log_data}")
            
            new_log = logActivityRepository.createLogActivity(log_data)
            
            if new_log and hasattr(new_log, 'log_id'):
                log_id = str(new_log.log_id)
                print(f"✅ Log berhasil dibuat dengan log_id: {log_id}")
                return log_id
            else:
                print("❌ Error: Log creation returned None or invalid object")
                return None
                
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Gagal membuat log: {str(e)}")
            return None

    def updateGrantAccess(self, log_id, grant_access):
        """
        Update grant_access di log_activity berdasarkan hasil prediksi.
        grant_access: Boolean (True = aman, False = berbahaya)
        """
        try:
            if not log_id:
                print("❌ Error: log_id tidak boleh kosong")
                return False
                
            print(f"📝 Attempting to update grant_access for log_id: {log_id} → {grant_access}")
            
            updated_log = logActivityRepository.updateGrantAccess(str(log_id), grant_access)
            
            if updated_log:
                print(f"✅ Grant access berhasil diupdate untuk log_id {log_id}: {grant_access}")
                return True
            else:
                print(f"⚠️ Log dengan log_id {log_id} tidak ditemukan")
                return False
                
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Gagal update grant_access: {str(e)}")
            return False


    def sendNotification(self, childId, predictId,parentId, url, logId):
        notif_url = "https://snailly-backend.codelabspace.or.id/notification/send"
        payload = {
            "childId": childId,
            "parentId": parentId,
        }

        if(logId is not None):
            payload["logId"] = logId

        print(f"Payload Send Notification {payload}")
        headers = {"Content-Type": "application/json"}

        try:
            res = requests.post(notif_url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
            print(f"Notifikasi berhasil dikirim: {data}")
        except Exception as e:
            traceback.print_exc()
            print(f"Gagal kirim notifikasi: {e}")

    def _load_model(self):
        MODEL_PATH = "public/models"
        def get_latest_file(prefix):
            files = [
                f for f in os.listdir(MODEL_PATH)
                if f.startswith(prefix) and f.endswith(".pkl")
            ]
            if not files:
                return None
            # Urutkan berdasarkan waktu modifikasi, ambil terbaru
            latest_file = max(
                files,
                key=lambda x: os.path.getmtime(os.path.join(MODEL_PATH, x))
            )
            return os.path.join(MODEL_PATH, latest_file)

        svm_path = get_latest_file("svm_model_") or os.path.join(MODEL_PATH, "svm_model.pkl")
        tfidf_path = get_latest_file("tfidf_vectorizer_") or os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")

        svm_model, tfidf_vectorizer = None, None

        try:
            if os.path.exists(svm_path):
                with open(svm_path, 'rb') as f:
                    svm_model = joblib.load(f)
                print(f"Model SVM dimuat dari {svm_path}")
            else:
                print(f"PERINGATAN: File model SVM tidak ditemukan di '{svm_path}'.")
        except Exception as e:
            print(f"ERROR saat memuat model SVM dari '{svm_path}': {e}")

        try:
            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    tfidf_vectorizer = joblib.load(f)
                print(f"Vectorizer TF-IDF dimuat dari {tfidf_path}")
            else:
                print(f"PERINGATAN: File vectorizer TF-IDF tidak ditemukan di '{tfidf_path}'.")
        except Exception as e:
            print(f"ERROR saat memuat vectorizer TF-IDF dari '{tfidf_path}': {e}")

        return svm_model, tfidf_vectorizer
    
    def createPredictData(self, data):
        try:
            if self.svm_model is None or self.tfidf_vectorizer is None:
                return self.failedOrSuccessRequest('failed', 500, "Model belum dimuat.")
            
            text = data.get('text', None)
            child_id = data.get("child_id")
            parent_id = data.get("parent_id")
            url = data.get("url")

            if not text:
                return self.failedOrSuccessRequest('failed', 404, 'No text provided')
            
            #============ 1. INSERT LOG ACTIVITY TERLEBIH DAHULU =============#
            print(f"Mengirim log untuk URL: {url}")
            log_id = self.sendLog(child_id, parent_id, url)
            
            if not log_id:
                error_msg = f"Gagal insert log_activity. child_id={child_id} TIDAK DITEMUKAN di database. User harus re-login atau data sudah dihapus."
                print(f"❌ {error_msg}")
                return self.failedOrSuccessRequest('failed', 400, error_msg)
            
            print(f"Log ID berhasil dibuat: {log_id}")
            
            #============ 2. PREDIKSI UNTUK TEKS FULL =============#
            print(f"Memulai prediksi untuk teks: {text[:50]}...")
            X = self.tfidf_vectorizer.transform([text])
            
            predicted_labels = self.svm_model.predict(X).tolist()
            predicted_proba = self.svm_model.predict_proba(X).tolist()
            
            proba_array = predicted_proba[0] if predicted_proba else [0.5, 0.5]  # [prob_aman, prob_berbahaya]
            
            # Simpan predict data
            predictData = predictDataRepository.createNewPredictData({
                "text": text,
                "label": predicted_labels[0],
                "predicted_proba": predicted_proba,
                "url": url,
                "parent_id": parent_id,
                "child_id": child_id,
                "log_id": log_id
            })
            predictDataDict = queryResultToDict([predictData])[0]
            
            #============ 3. UPDATE GRANT_ACCESS BERDASARKAN HASIL PREDIKSI =============#
            grant_access = True if predicted_labels[0] == 'aman' else False  
            update_success = self.updateGrantAccess(log_id, grant_access)
            
            if not update_success:
                print(f"WARNING: Gagal update grant_access untuk log_id {log_id}")

            #============ 4. PREDIKSI PER SEGMENT =============#
            clean_record = cleanDataRepository.getCleanDataByUrl(url)
            if clean_record and isinstance(clean_record.segments, list) and clean_record.segments:
                new_segments = []
                for seg in clean_record.segments:
                    transcript = seg.get('transcript', '').strip()
                    visual = seg.get('visual', '').strip()
                    combined_text = f"{transcript} {visual}".strip()

                    if combined_text == "":
                        seg["danger"] = "kosong"
                    else:
                        X_seg = self.tfidf_vectorizer.transform([combined_text])
                        y_pred = self.svm_model.predict(X_seg)[0]
                        seg["danger"] = "bahaya" if y_pred == 1 else "aman"

                    new_segments.append(seg)

                cleanDataRepository.updateCleanData(
                    clean_record.clean_data_id, {"segments": new_segments}
                )
                print(f"Prediksi per segment (combined) telah diperbarui di database untuk URL {url}.")

            #============ NOTIFIKASI URL =============#
            existing_url_classification = urlClassificationRepository.getUrlClassificationByUrl(data.get('url', None))
            print(f"Existing URL classification: {existing_url_classification}")
            if not existing_url_classification:
                parsed = urlparse(url)
                hostname = parsed.hostname
                predict_id = predictDataDict.get('id', None)
                print(f"Predict ID: {predict_id}")
                print(f"{data.get('url', None)} tidak ada di database, SEND NOTIFICATION")
                self.sendNotification(
                    childId=child_id, 
                    predictId=predict_id,
                    parentId=parent_id,
                    url=hostname,
                    logId=log_id
                )

            # ✅ Return response dengan mapping yang benar
            return self.failedOrSuccessRequest('success', 201, {
                "log_id": log_id,
                "labels": predicted_labels[0],
                "probabilities": proba_array,  
                "grant_access": grant_access   
            })
        except Exception as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))
    
    def getMajorityLabel(self):
        try:
            predict_data = predictDataRepository.getAllPredictData()
            # Pastikan predict_data adalah list dict
            if predict_data and not isinstance(predict_data[0], dict):
                predict_data = queryResultToDict(predict_data)

            if not predict_data:
                return "No prediction data found."

            url_groups = {}
            for item in predict_data:
                url = item.get('url')
                label = item.get('label')
                if not url:
                    continue
                if url not in url_groups:
                    url_groups[url] = []
                url_groups[url].append(label)

            majority_labels = {}
            for url, labels in url_groups.items():
                if len(labels) % 2 == 1:  # jumlah ganjil → ambil majority
                    majority_label = max(set(labels), key=labels.count)
                    majority_labels[url] = majority_label
                else:
                    print(f"URL {url} punya jumlah label genap ({len(labels)}), dilewati.")

            return majority_labels

        except Exception as e:
            return str(e)
            
    def createRetrainModel(self):
        try:
            # Ambil majority label dari predict data
            majority_labels = self.getMajorityLabel()
            print(f"Majority labels: {majority_labels}")

            majority_data = majority_labels  # dict {url: label}
            print(f"Majority data: {majority_data}")
            # Masukkan majority label yang belum ada di urlClassification ke database urlClassification
            if isinstance(majority_data, dict):
                for url, label in majority_data.items():
                    print(f"Memeriksa URL: {url} dengan label: {label}")
                    existing = urlClassificationRepository.getUrlClassificationByUrl(url)
                    print(f"Existing URL classification: {existing}")
                    if not existing:
                        print(f"Menambahkan URL baru ke urlClassification: {url}")
                        clean_data_record = cleanDataRepository.getCleanDataByUrl(url)
                        if clean_data_record:
                            # 3a. Jika ditemukan, lanjutkan proses
                            print(f"Data teks ditemukan. Menambahkan ke urlClassification...")
                            stopwords = clean_data_record.stopword_removed_tokens
                            urlClassificationRepository.createNewUrlClassification({
                                'url': url,
                                'label': label,
                                'stopword_removed_tokens': stopwords
                            })
                            
                            # Setelah data dipindahkan ke dataset training final, hapus dari data prediksi sementara
                            delete = predictDataRepository.deletePredictDataByUrl(url)
                            print(f"Data prediksi sementara untuk URL {url} telah dihapus: {delete}")
                        else:
                            # 3b. Jika TIDAK ditemukan, beri peringatan dan lewati URL ini
                            print(f"PERINGATAN: Tidak ditemukan data teks di 'clean_data' untuk URL {url}. "
                                f"URL ini akan dilewati dan tidak ditambahkan ke dataset training.")
                            # 'continue' tidak wajib, karena ini akhir dari blok if, tapi memperjelas alur
                            continue 
            # Ambil ulang data urlClassification yang sudah lengkap
            data_url = urlClassificationRepository.getAllUrlClassifications()
            print(data_url)
            df_url_classification = pd.DataFrame(queryResultToDict(data_url))

            if df_url_classification.empty:
                return self.failedOrSuccessRequest('failed', 400, "'url' column missing in urlClassification data or data kosong")

            print(f"Data urlClassification untuk retrain: {df_url_classification.head()}")

            # df_majority = pd.DataFrame(majority_data.items(), columns=['url', 'label'])

            # # Merge data urlClassification dan majority label berdasarkan 'url'
            print(df_url_classification.describe())

            X_raw = df_url_classification['stopword_removed_tokens']
            y = df_url_classification['label']

            # --- PERBAIKAN UTAMA DI SINI ---
            # Fungsi helper untuk mengubah list/string-of-list menjadi string tunggal
            def join_tokens(doc):
                # Periksa jika data adalah string yang terlihat seperti list, e.g., "['a', 'b']"
                if isinstance(doc, str) and doc.startswith('[') and doc.endswith(']'):
                    try:
                        # Ubah string menjadi list asli dengan aman
                        actual_list = ast.literal_eval(doc)
                        return ' '.join(actual_list)
                    except (ValueError, SyntaxError):
                        # Jika gagal di-parse, kembalikan string kosong agar bisa di-handle
                        return ""
                # Jika sudah berupa list, gabungkan
                elif isinstance(doc, list):
                    return ' '.join(doc)
                # Jika sudah string biasa (mungkin dari kasus lain)
                elif isinstance(doc, str):
                    return doc
                # Jika format lain atau NaN, kembalikan string kosong
                return ""

            print("Menggabungkan token menjadi string untuk TfidfVectorizer...")
            X = X_raw.apply(join_tokens)
            
            # Hapus NaN
            valid_idx = ~(X.isnull() | y.isnull())
            X, y = X[valid_idx], y[valid_idx]

            if len(X) < 10:
                return self.failedOrSuccessRequest('failed', 400, "Data terlalu sedikit untuk training.")

            # Split data
            try:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.4, stratify=y, random_state=42
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
                )
            except Exception:
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # TF-IDF
            tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_val_tfidf   = tfidf.transform(X_val)
            X_test_tfidf  = tfidf.transform(X_test)

            # Train model
            svm_model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
            svm_model.fit(X_train_tfidf, y_train)

            # Evaluasi
            val_acc = accuracy_score(y_val, svm_model.predict(X_val_tfidf))
            test_acc = accuracy_score(y_test, svm_model.predict(X_test_tfidf))

            # Save model
            MODEL_PATH = "public/models"
            os.makedirs(MODEL_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(MODEL_PATH, f"svm_model_{timestamp}.pkl")
            vectorizer_filename = os.path.join(MODEL_PATH, f"tfidf_vectorizer_{timestamp}.pkl")

            joblib.dump(svm_model, model_filename)
            joblib.dump(tfidf, vectorizer_filename)
            joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_model.pkl"))
            joblib.dump(tfidf, os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))

            # Save summary
            summary = {
                'timestamp': timestamp,
                'validation_accuracy': val_acc,
                'test_accuracy': test_acc,
                'num_features': X_train_tfidf.shape[1],
                'num_samples': len(X),
                'num_classes': len(y.unique()),
                'classes': list(svm_model.classes_)
            }
            with open(os.path.join(MODEL_PATH, "training_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # Update model in memory
            self.svm_model = svm_model
            self.tfidf_vectorizer = tfidf

            return self.failedOrSuccessRequest('success', 200, summary)

        except Exception as e:
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))
    
    def predictLabel(self, data):
        """
        Prediksi label dari teks saja, return string label.
        data: dict, minimal berisi key 'text'
        """
        if self.svm_model is None or self.tfidf_vectorizer is None:
            return ""
        text = data.get('text', None)
        if not text:
            return ""
        try:
            X = self.tfidf_vectorizer.transform([text])
            predicted_label = self.svm_model.predict(X)[0]
            return str(predicted_label)
        except Exception as e:
            print("Error prediksi label:", e)
            return ""
    
    def predictLabelBERT(self, text):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            print("transformers dan torch belum terinstall.")
            return ""
        model_dir = os.path.join("public", "models", "xlm-roberta-weighted")
        if not hasattr(self, "xlm_tokenizer") or self.xlm_tokenizer is None:
            try:
                self.xlm_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            except Exception as e:
                print("Error loading XLM-Roberta tokenizer:", e)
                self.xlm_tokenizer = None
        if not hasattr(self, "xlm_model") or self.xlm_model is None:
            try:
                self.xlm_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            except Exception as e:
                print("Error loading XLM-Roberta model:", e)
                self.xlm_model = None
        if self.xlm_tokenizer is None or self.xlm_model is None:
            print("XLM-Roberta model/tokenizer not loaded.")
            return ""
        if not isinstance(text, str) or not text.strip():
            return ""
        try:
            inputs = self.xlm_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.xlm_model(**inputs)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()
            return str(predicted_label)
        except Exception as e:
            print("Error prediksi label XLM-Roberta:", e)
            return ""
