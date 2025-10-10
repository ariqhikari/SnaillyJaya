
import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "public/models/mixtext_bert"

# --- Load BERT/MixText ---
def load_bert(model_path=MODEL_PATH, hf_model="bert-base-multilingual-cased"):
    """
    Memuat model BERT/MixText dari lokal jika ada, jika tidak unduh dari Hugging Face.
    Return: tokenizer, model
    """
    try:
        if os.path.exists(model_path) and os.listdir(model_path):
            print("🔄 Memuat BERT/MixText dari lokal...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(DEVICE)
        else:
            print("⬇️ Mengunduh BERT/MixText dari Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModel.from_pretrained(hf_model).to(DEVICE)

            # Simpan ke lokal
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)

        return tokenizer, model
    except Exception as e:
        print(f"❌ Gagal memuat BERT: {e}")
        return None, None


# --- Caption → Embedding ---
def caption_to_embedding(caption: str, tokenizer, model):
    """
    Mengubah caption menjadi embedding BERT (output: numpy array 1D)
    """
    if tokenizer is None or model is None:
        return None

    if not isinstance(caption, str) or not caption.strip():
        return None

    try:
        tokens = tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)

        # Ambil CLS token → representasi kalimat
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"❌ Gagal membuat embedding untuk caption '{caption[:30]}...': {e}")
        return None


# --- Integrasi dengan caption_images_in_folder ---
def process_captions_with_bert(folder_path: str):
    """
    Proses folder gambar: captioning → embedding BERT untuk setiap caption
    Return: DataFrame dengan kolom [File, Caption, Embedding]
    """
    from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder

    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        return pd.DataFrame()

    # 1. Captioning
    df = caption_images_in_folder(folder_path)
    if df is None or df.empty:
        print("⚠️ Tidak ada caption yang bisa diproses.")
        return pd.DataFrame()

    # 2. Load BERT
    tokenizer, model = load_bert()
    if tokenizer is None or model is None:
        return pd.DataFrame()

    # 3. Konversi setiap caption → embedding
    df["Embedding"] = df["Caption"].apply(
        lambda cap: caption_to_embedding(cap, tokenizer, model)
    )

    return df


# --- Pemakaian ---
if __name__ == "__main__":
    folder = "folder_gambar"  # ganti sesuai path gambar kamu
    result_df = process_captions_with_bert(folder)
    print(result_df.head())

# import torch
# import os
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel

# # --- Config ---
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "public/models/mixtext_bert"

# # --- Load BERT/MixText ---
# def load_bert(model_path=MODEL_PATH, hf_model="bert-base-multilingual-cased"):
#     """
#     Memuat model BERT/MixText dari lokal jika ada, jika tidak unduh dari Hugging Face.
#     Return: tokenizer, model
#     """
#     if os.path.exists(model_path):
#         print("🔄 Memuat BERT/MixText dari lokal...")
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModel.from_pretrained(model_path).to(DEVICE)
#     else:
#         print("⬇️ Mengunduh BERT/MixText dari Hugging Face...")
#         tokenizer = AutoTokenizer.from_pretrained(hf_model)
#         model = AutoModel.from_pretrained(hf_model).to(DEVICE)
#         tokenizer.save_pretrained(model_path)
#         model.save_pretrained(model_path)
#     return tokenizer, model

# # --- Caption → Embedding ---
# def caption_to_embedding(caption: str, tokenizer, model):
#     """
#     Mengubah caption menjadi embedding BERT (output: numpy array 1D)
#     """
#     if not isinstance(caption, str) or not caption.strip():
#         return None
#     tokens = tokenizer(caption, truncation=True, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outputs = model(**tokens)
#     # Ambil CLS token, simpan sebagai list
#     embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
#     return embedding

# # --- Integrasi dengan caption_images_in_folder ---
# def process_captions_with_bert(folder_path: str):
#     """
#     Proses folder gambar: captioning → embedding BERT untuk setiap caption
#     Return: DataFrame dengan kolom Caption dan Embedding
#     """
#     from src.utils.Preprocessing.ImageProcessor import caption_images_in_folder
#     # 1. Captioning
#     df = caption_images_in_folder(folder_path)
#     if df.empty:
#         print("Tidak ada caption yang bisa diproses.")
#         return pd.DataFrame()

#     # 2. Load BERT
#     tokenizer, model = load_bert()

#     # 3. Konversi setiap caption → embedding (list supaya aman di DB/JSON)
#     def safe_embed(cap):
#         try:
#             emb = caption_to_embedding(cap, tokenizer, model)
#             return emb.tolist() if emb is not None else None
#         except Exception as e:
#             print(f"Gagal embedding caption: {cap[:30]}... | Error: {e}")
#             return None

#     df["Embedding"] = df["Caption"].apply(safe_embed)

#     return df

# # --- Pemakaian ---
# if __name__ == "__main__":
#     folder = "folder_gambar"
#     result_df = process_captions_with_bert(folder)
#     print(result_df.head())
