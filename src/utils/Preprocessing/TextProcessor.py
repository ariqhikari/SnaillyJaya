import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag, word_tokenize

# --- Pastikan semua resource NLTK tersedia ---
REQUIRED_NLTK_PACKAGES = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng"
]

for pkg in REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        print(f"🔽 Mengunduh NLTK data: {pkg} ...")
        nltk.download(pkg, quiet=True)

# --- Inisialisasi Komponen NLP (dilakukan sekali saat modul di-load) ---
try:
    print("Menginisialisasi komponen TextProcessor...")
    ID_STEMMER = StemmerFactory().create_stemmer()
    EN_LEMMATIZER = WordNetLemmatizer()
    STOPWORD_ID = set(stopwords.words("indonesian"))
    STOPWORD_EN = set(stopwords.words("english"))
    STOPWORD_ALL = STOPWORD_ID.union(STOPWORD_EN)
    print("✅ Komponen TextProcessor siap.")
except Exception as e:
    print(f"❌ Gagal inisialisasi komponen NLP. Error: {e}")
    print("   Jalankan manual: python -m nltk.downloader all")

# --- Fungsi Bantuan ---
def get_wordnet_pos(token):
    """Map POS tag ke format yang diterima oleh WordNetLemmatizer."""
    tag = pos_tag([token])[0][1]
    if tag.startswith("J"): return wordnet.ADJ
    elif tag.startswith("V"): return wordnet.VERB
    elif tag.startswith("N"): return wordnet.NOUN
    elif tag.startswith("R"): return wordnet.ADV
    else: return wordnet.NOUN

def stem_bilingual(tokens):
    """Melakukan lemmatization untuk kata Inggris dan stemming untuk kata Indonesia."""
    out = []
    for t in tokens:
        lemma = EN_LEMMATIZER.lemmatize(t, pos=get_wordnet_pos(t))
        if lemma != t:  # Jika kata berubah (berarti itu kata Inggris yang dilemmatize)
            out.append(lemma)
        else:  # Jika tidak, anggap kata Indonesia dan stem
            out.append(ID_STEMMER.stem(t))
    return out

# --- Fungsi Utama Preprocessing ---
def process_text(raw_text: str):
    """
    Membersihkan teks mentah dari hasil scraping melalui tahapan preprocessing.
    Mengembalikan teks bersih dan token hasil proses.
    """
    if not isinstance(raw_text, str):
        return ""

    # 1. Case Folding
    text = raw_text.lower()
    # 2. Menghapus angka
    text = re.sub(r"\d+", "", text)
    # 3. Menghapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)
    # 4. Menghapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenization
    tokens = word_tokenize(text)

    # 6. Bilingual Stemming & Lemmatization
    stemmed_tokens = stem_bilingual(tokens)

    # 7. Stopword Removal
    final_tokens = [token for token in stemmed_tokens if token not in STOPWORD_ALL and len(token) > 1]

    clean_text = " ".join(final_tokens)
    return clean_text, final_tokens
