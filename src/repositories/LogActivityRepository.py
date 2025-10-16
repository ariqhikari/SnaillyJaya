from src.server.main import db, main_app
from src.config.database import generateDatabase
from src.models.LogActivity import log_activity
from datetime import datetime
import traceback

class LogActivityRepository:
    def createLogActivity(self, data):
        """
        Insert new log_activity record
        
        Args:
            data (dict): Dictionary containing childId, url, and optional fields
                - childId (required)
                - url (required)
                - parentId (optional)
                - grant_access (optional)
        
        Returns:
            log_activity: The created log_activity object
        """
        try:
            # Validasi field wajib
            if not data.get('childId') or not data.get('url'):
                raise ValueError("childId dan url wajib diisi")
            
            new_log = log_activity(
                childId=str(data['childId']),
                url=str(data['url']),
                parentId=str(data.get('parentId')) if data.get('parentId') else None,
                grant_access=None
            )
            
            db.session.add(new_log)
            db.session.commit()
            db.session.refresh(new_log)
            
            print(f"✅ Log berhasil dibuat: {new_log.log_id}")
            return new_log
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error saat create log activity:")
            traceback.print_exc()
            raise e
    
    def updateGrantAccess(self, log_id, grant_access):
        """
        Update grant_access field for a specific log_activity
        
        Args:
            log_id (str): The UUID of the log_activity record
            grant_access (bool): True if safe, False if dangerous
        
        Returns:
            log_activity: The updated log_activity object, or None if not found
        """
        try:
            log = db.session.query(log_activity).filter(
                log_activity.log_id == str(log_id)
            ).first()
            
            if not log:
                print(f"⚠️ Log dengan log_id {log_id} tidak ditemukan")
                return None
            
            log.grant_access = grant_access
            log.updatedAt = datetime.now()
            
            db.session.commit()
            db.session.refresh(log)
            
            print(f"✅ Grant access updated: {log_id} → {grant_access}")
            return log
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error saat update grant_access:")
            traceback.print_exc()
            raise e
    
    def getLogById(self, log_id):
        """
        Get log_activity by log_id
        
        Args:
            log_id (str): The UUID of the log_activity record
        
        Returns:
            log_activity: The log_activity object, or None if not found
        """
        try:
            log = db.session.query(log_activity).filter(
                log_activity.log_id == str(log_id)
            ).first()
            return log
        except Exception as e:
            print(f"❌ Error saat get log by id:")
            traceback.print_exc()
            return None
    
    def getLogsByChildId(self, child_id, limit=100):
        """
        Get all logs by child_id
        
        Args:
            child_id (str): The child ID
            limit (int): Maximum number of logs to return
        
        Returns:
            list: List of log_activity objects
        """
        try:
            logs = db.session.query(log_activity).filter(
                log_activity.childId == str(child_id)
            ).order_by(log_activity.createdAt.desc()).limit(limit).all()
            return logs
        except Exception as e:
            print(f"❌ Error saat get logs by child_id:")
            traceback.print_exc()
            return []
