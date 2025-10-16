from src.server.main import db, main_app
from src.config.database import generateDatabase
import uuid

class log_activity(db.Model):
    __tablename__ = 'log_activity'
    
    log_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parentId = db.Column(db.String(255), nullable=True)
    childId = db.Column(db.String(255), nullable=False)
    url = db.Column(db.Text, nullable=False)
    web_title = db.Column(db.Text, nullable=True)
    detail_url = db.Column(db.Text, nullable=True)
    web_description = db.Column(db.Text, nullable=True)
    grant_access = db.Column(db.Boolean, nullable=True)
    createdAt = db.Column(db.DateTime, server_default=db.func.now())
    updatedAt = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
    deletedAt = db.Column(db.DateTime, nullable=True)

    def __init__(self, childId, url, parentId=None, web_title=None, detail_url=None, web_description=None, grant_access=None):
        self.log_id = str(uuid.uuid4())
        self.childId = childId
        self.url = url
        self.parentId = parentId
        self.web_title = web_title
        self.detail_url = detail_url
        self.web_description = web_description
        self.grant_access = grant_access
