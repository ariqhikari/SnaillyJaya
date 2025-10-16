from src.server.main import db, main_app
from src.config.database import generateDatabase
from src.models.LogActivity import log_activity

class LogActivityRepository:
    def createLogActivity(self, data):
        """
        Insert new log_activity record
        
        Args:
            data (dict): Dictionary containing childId, url, and optional fields
                - childId (required)
                - url (required)
                - parentId (optional)
                - web_title (optional)
                - detail_url (optional)
                - web_description (optional)
                - grant_access (optional)
        
        Returns:
            log_activity: The created log_activity object
        """
        new_log = log_activity(
            childId=data['childId'],
            url=data['url'],
            parentId=data.get('parentId'),
            web_title=data.get('web_title'),
            detail_url=data.get('detail_url'),
            web_description=data.get('web_description'),
            grant_access=data.get('grant_access')
        )
        
        db.session.add(new_log)
        db.session.commit()
        
        return new_log
    
    def updateGrantAccess(self, log_id, grant_access):
        """
        Update grant_access field for a specific log_activity
        
        Args:
            log_id (str): The UUID of the log_activity record
            grant_access (bool): True if safe, False if dangerous
        
        Returns:
            log_activity: The updated log_activity object, or None if not found
        """
        log = log_activity.query.filter_by(log_id=log_id).first()
        
        if log:
            log.grant_access = grant_access
            db.session.commit()
            return log
        
        return None
    
    def getLogById(self, log_id):
        """
        Get log_activity by log_id
        
        Args:
            log_id (str): The UUID of the log_activity record
        
        Returns:
            log_activity: The log_activity object, or None if not found
        """
        return log_activity.query.filter_by(log_id=log_id).first()
