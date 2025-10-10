from src.server.main import db,main_app
from src.config.database import generateDatabase
from sqlalchemy.dialects.postgresql import JSON

class url_classification(db.Model):
  __tablename__ = 'url_classification'
  
  id = db.Column(db.Integer, primary_key=True)
  url = db.Column(db.Text, unique=True)
  label = db.Column(db.String(255), nullable=False)
  stopword_removed_tokens =  db.Column(JSON, nullable=False)


  def __init__(self, url,  label, stopword_removed_tokens):
    self.url = url
    self.label = label
    self.stopword_removed_tokens = stopword_removed_tokens
