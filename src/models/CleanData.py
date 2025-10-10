from src.server.main import db,main_app
from src.config.database import generateDatabase
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import ARRAY
class clean_data(db.Model):
  __tablename__ = 'clean_data'
  
  clean_data_id = db.Column(db.Integer, primary_key=True)
  url = db.Column(db.Text, unique=True, index=True)
  text = db.Column(db.Text, nullable=False)
  stopword_removed_tokens =  db.Column(JSON, nullable=False)
  raw_text = db.Column(db.Text, nullable=False)
  link_gambar = db.Column(db.Text)
  folder_gambar = db.Column(db.Text)
  link_video = db.Column(db.Text)
  folder_video = db.Column(db.Text)
  segments = db.Column(JSON, nullable=True)

  def __init__(self, url, text, raw_text, stopword_removed_tokens, link_gambar, folder_gambar, link_video, folder_video, segments=None):
    self.url = url
    self.text = text
    self.raw_text = raw_text
    self.stopword_removed_tokens = stopword_removed_tokens
    self.link_gambar = link_gambar
    self.folder_gambar = folder_gambar
    self.link_video = link_video
    self.folder_video = folder_video
    self.segments = segments or []