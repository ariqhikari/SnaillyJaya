from src.models.ScreenshotData import screenshot_data, db

class ScreenshotDataRepository:
    def getAllScreenshots(self):
        return screenshot_data.query.all()

    def getScreenshotById(self, screenshot_id):
        return screenshot_data.query.filter_by(screenshot_id=screenshot_id).first()

    def createNewScreenshotData(self, data):
        required_keys = {'text', 'raw_text', 'stopword_removed_tokens', 'folder_gambar','label'}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Data tidak lengkap untuk membuat ScreenshotData. Butuh: {required_keys}")

        new_screenshot = screenshot_data(
            text=data['text'],
            raw_text=data['raw_text'],
            stopword_removed_tokens=data['stopword_removed_tokens'],
            folder_gambar=data['folder_gambar'],
            label=data.get('label') #inikan penambahan baru
        )
        db.session.add(new_screenshot)
        db.session.commit()
        return new_screenshot

    def updateScreenshotData(self, id, data, merge_json=True):
        record = screenshot_data.query.filter_by(screenshot_id=id).first()
        if not record:
            return False

        for key, value in data.items():
            if hasattr(record, key):
                if merge_json and isinstance(getattr(record, key), list) and isinstance(value, list):
                    setattr(record, key, getattr(record, key) + value)
                elif merge_json and isinstance(getattr(record, key), dict) and isinstance(value, dict):
                    merged = {**getattr(record, key), **value}
                    setattr(record, key, merged)
                else:
                    setattr(record, key, value)

        db.session.commit()
        return record

    def deleteScreenshotData(self, id):
        record = screenshot_data.query.filter_by(screenshot_id=id).first()
        if not record:
            return False
        db.session.delete(record)
        db.session.commit()
        return True
