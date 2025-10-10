from main import main_app, PORT
# from src.utils.Preprocessing.TextProcessorBERT import load_bert

# Muat model BERT saat server start
# load_bert()
if __name__ == '__main__':
    main_app.run(host="0.0.0.0", port=PORT, debug=True, threaded=True,)
