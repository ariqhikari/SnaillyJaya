from src.server.main import db, main_app
from src.config.database import generateDatabase
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

class log_activity(db.Model):
    __tablename__ = 'log_activity'
    
    log_id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parentId = db.Column(db.String(255), nullable=True)
    childId = db.Column(db.String(255), nullable=False)
    url = db.Column(db.Text, nullable=False)
    grant_access = db.Column(db.Boolean, nullable=True)
    createdAt = db.Column(db.DateTime, default=datetime.now, nullable=False)
    updatedAt = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)
    deletedAt = db.Column(db.DateTime, nullable=True)

    def __init__(self, childId, url, parentId=None, grant_access=None):
        self.log_id = str(uuid.uuid4())
        self.childId = childId
        self.url = url
        self.parentId = parentId
        self.grant_access = grant_access
        self.createdAt = datetime.now()
        self.updatedAt = datetime.now()
    
    def __repr__(self):
        return f'<LogActivity {self.log_id} - {self.url}>'
    
    def to_dict(self):
        return {
            'log_id': str(self.log_id),
            'parentId': self.parentId,
            'childId': self.childId,
            'url': self.url,
            'grant_access': self.grant_access,
            'createdAt': self.createdAt.isoformat() if self.createdAt else None,
            'updatedAt': self.updatedAt.isoformat() if self.updatedAt else None,
            'deletedAt': self.deletedAt.isoformat() if self.deletedAt else None,
        }
