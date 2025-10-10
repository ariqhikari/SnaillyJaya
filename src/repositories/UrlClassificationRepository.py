from src.models.UrlClassification import url_classification,db
from src.repositories.CleanDataRepository import CleanDataRepository
import pandas as pd
import ast # Tambahkan import ini untuk konversi string ke list

cleanDataRepository = CleanDataRepository()

class UrlClassificationRepository:
  def getAllUrlClassifications(self):
    return url_classification.query.all()
  def createNewUrlClassification(self, data):
    new_url_classification = url_classification(
      url=data['url'],
      label=data['label'],
      stopword_removed_tokens=data['stopword_removed_tokens']
    )
    db.session.add(new_url_classification)
    db.session.commit()
    return new_url_classification
  def getUrlClassificationById(self, id):
    return url_classification.query.get(id)
  def getUrlClassificationByUrl(self, url):
    return url_classification.query.filter_by(url=url).first()
  def updateUrlClassification(self, id, data):
    url_classification_to_update = url_classification.query.get(id)
    if not url_classification_to_update:
      return None
    url_classification_to_update.url = data['url']
    url_classification_to_update.label = data['label']
    db.session.commit()
    return url_classification_to_update

  def deleteUrlClassification(self, id):
    url_classification_to_delete = url_classification.query.get(id)
    if not url_classification_to_delete:
      return None
    db.session.delete(url_classification_to_delete)
    db.session.commit()
    return url_classification_to_delete

  def createSeedDataset(self, seed_dataframe: pd.DataFrame):
      new_records_count = 0
      print(seed_dataframe)
      for index, row in seed_dataframe.iterrows():
          url = row.get('url')
          if not url:
              continue # Lewati baris jika tidak ada URL
          existing_record = self.getUrlClassificationByUrl(url)
          if existing_record:
              print(f"Melewatkan (sudah ada): {url}")
              continue
          tokens_str = row.get('stopword_removed_tokens')
          new_record = url_classification(
              url=url,
              label=row.get('label'),
              stopword_removed_tokens=tokens_str
          )
          db.session.add(new_record)
          new_records_count += 1
      
      if new_records_count > 0:
          print(f"Menyimpan {new_records_count} data baru ke database...")
          db.session.commit()
      
      return new_records_count