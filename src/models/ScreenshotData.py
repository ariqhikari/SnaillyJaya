from src.server.main import db, main_app
from src.config.database import generateDatabase
from sqlalchemy.dialects.postgresql import JSON

class screenshot_data(db.Model):
    __tablename__ = 'screenshot_data'
  
    screenshot_id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    stopword_removed_tokens = db.Column(JSON, nullable=False)
    folder_gambar = db.Column(db.Text, nullable=False)
    label = db.Column(db.Text, nullable=False) #jangan lupa ditambahin di migration

    def __init__(self, text, raw_text, stopword_removed_tokens, folder_gambar, label):
        self.text = text
        self.raw_text = raw_text
        self.stopword_removed_tokens = stopword_removed_tokens
        self.folder_gambar = folder_gambar
        self.label = label 
