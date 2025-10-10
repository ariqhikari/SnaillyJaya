
from src.repositories.CleanDataRepository import CleanDataRepository
from src.repositories.PredictDataRepository import PredictDataRepository
from src.repositories.UrlClassificationRepository import UrlClassificationRepository
from src.utils.convert import queryResultToDict
from src.services.Service import Service
from src.utils.errorHandler import errorHandler
import os # Pastikan os diimpor
import pandas as pd
import ast

cleanDataRepository = CleanDataRepository()
predictDataRepository = PredictDataRepository()
urlClassificationRepository = UrlClassificationRepository()
class HITLService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }

    def updatePredictLabelById(self, predict_id, new_label):
        try:
            # Validasi dan proses data di sini
            updated_predict_data = predictDataRepository.updatePredictLabelById(predict_id, new_label)
            if not updated_predict_data:
                return self.failedOrSuccessRequest('failed', 404, 'Data tidak ditemukan')
            return self.failedOrSuccessRequest('success', 200, 'Data berhasil diperbarui')
        except Exception as e:
            return self.failedOrSuccessRequest('failed', 500, str(e))

    def updatePredictLabelByLogId(self, log_id, new_label):
        updated_predict_data = predictDataRepository.updatePredictLabelByLogId(log_id, new_label)
        if not updated_predict_data:
            return self.failedOrSuccessRequest('failed', 404, 'Data tidak ditemukan')
        return self.failedOrSuccessRequest('success', 200, 'Data berhasil diperbarui')

    def createSeedDataset(self):
        try:
            # Langkah 1: Baca data dari CSV
            seed_dataframe = self.generateSeedData()
            if seed_dataframe.empty:
                return self.failedOrSuccessRequest('failed', 404, 'File dataset seed tidak ditemukan atau kosong.')
            
            newly_added_count = urlClassificationRepository.createSeedDataset(seed_dataframe)
            
            response_data = {
                'message': 'Seeding process completed.',
                'new_records_added': newly_added_count
            }
            return self.failedOrSuccessRequest('success', 201, response_data)
        
        except FileNotFoundError:
            return self.failedOrSuccessRequest('failed', 404, f"File dataset seed tidak ditemukan di path: public/dataset_final.csv")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, str(e))
    
    def generateSeedData(self):
        # Logika ini sudah benar
        SEED_PATH = "public/dataset_final.csv"
        if not os.path.exists(SEED_PATH):
            raise FileNotFoundError()
        seed_data = pd.read_csv(SEED_PATH)
        if "stopword_removed_tokens" in seed_data.columns:
            seed_data["stopword_removed_tokens"] = seed_data["stopword_removed_tokens"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        return seed_data