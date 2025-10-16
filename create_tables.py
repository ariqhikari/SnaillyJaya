"""
Script untuk membuat semua table di database.
Jalankan dengan: python create_tables.py
"""
from src.server.main import main_app, db

# Import semua models agar SQLAlchemy tahu tentang mereka
from src.models.CleanData import clean_data
from src.models.PredictData import predict_data
from src.models.ScreenshotData import screenshot_data
from src.models.UrlClassification import url_classification
from src.models.LogActivity import log_activity  # ← Import model baru

print("=" * 60)
print("Creating all database tables...")
print("=" * 60)

with main_app.app_context():
    # Drop all tables (hati-hati di production!)
    # db.drop_all()
    # print("✅ All tables dropped")
    
    # Create all tables
    db.create_all()
    print("✅ All tables created successfully!")
    
    # Print table names
    print("\n📊 Created tables:")
    for table in db.metadata.sorted_tables:
        print(f"   - {table.name}")
    
print("\n" + "=" * 60)
print("Done! You can now run the Flask server.")
print("=" * 60)
