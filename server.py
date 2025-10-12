import os
import threading
import time
from main import main_app, PORT

# --- Global State untuk Model Loading ---
models_loaded = False
models_loading = False
model_load_error = None
_load_start_time = None


def load_all_models_in_background():
    """
    Memuat semua models (BERT dan Image Captioning) di background thread.
    Fungsi ini tidak akan blocking server startup.
    """
    global models_loaded, models_loading, model_load_error, _load_start_time
    
    models_loading = True
    _load_start_time = time.time()
    print("🚀 Memulai loading models di background thread...")
    
    try:
        # Load BERT Model
        print("📥 Loading BERT model...")
        from src.utils.Preprocessing.TextProcessorBERT import load_bert
        tokenizer, bert_model = load_bert()
        if tokenizer is None or bert_model is None:
            raise Exception("BERT model gagal dimuat")
        print("✅ BERT model berhasil dimuat!")
        
        # Load Image Captioning Model
        print("📥 Loading Image Captioning model...")
        from src.utils.Preprocessing.ImageProcessor import load_image_captioning_model
        processor, blip_model = load_image_captioning_model()
        if processor is None or blip_model is None:
            raise Exception("Image Captioning model gagal dimuat")
        print("✅ Image Captioning model berhasil dimuat!")
        
        # Semua model berhasil dimuat
        models_loaded = True
        elapsed = time.time() - _load_start_time
        print(f"🎉 Semua models berhasil dimuat dalam {elapsed:.2f} detik!")
        
    except Exception as e:
        model_load_error = str(e)
        print(f"❌ Error saat memuat models: {e}")
        models_loaded = False
    finally:
        models_loading = False


# --- Health Check Endpoints ---
@main_app.route('/health')
def health_check():
    """
    Basic health check endpoint untuk Cloud Run.
    Return 200 jika server hidup, tidak peduli status model.
    """
    return {
        "status": "healthy",
        "service": "SnaillyJaya",
        "timestamp": time.time()
    }, 200


@main_app.route('/health/ready')
def readiness_check():
    """
    Readiness check endpoint untuk Cloud Run.
    Return 200 hanya jika models sudah siap digunakan.
    """
    if models_loaded:
        elapsed = time.time() - _load_start_time if _load_start_time else 0
        return {
            "status": "ready",
            "models_loaded": True,
            "load_time_seconds": round(elapsed, 2)
        }, 200
    elif models_loading:
        return {
            "status": "loading",
            "models_loaded": False,
            "message": "Models sedang dimuat di background"
        }, 503  # Service Unavailable
    elif model_load_error:
        return {
            "status": "error",
            "models_loaded": False,
            "error": model_load_error
        }, 503
    else:
        return {
            "status": "not_started",
            "models_loaded": False,
            "message": "Model loading belum dimulai"
        }, 503


if __name__ == '__main__':
    # Ambil environment variables untuk Cloud Run compatibility
    port = int(os.environ.get('PORT', PORT))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Starting server on 0.0.0.0:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Start background thread untuk model loading
    # Ini TIDAK akan blocking server startup
    model_thread = threading.Thread(target=load_all_models_in_background, daemon=True)
    model_thread.start()
    print("✅ Background model loading thread started!")
    
    # Start Flask server - ini akan langsung jalan tanpa menunggu models
    main_app.run(
        host="0.0.0.0",  # Required untuk Cloud Run
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader di production untuk avoid double loading
    )

