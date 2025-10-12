# Gunakan image Python 3.13 sesuai requirement
FROM python:3.13-slim

# Install dependencies system yang diperlukan
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy file dependency management
COPY pyproject.toml uv.lock ./

# Install dependencies menggunakan uv
RUN uv sync --frozen --no-dev

# Download NLTK data (simple, satu baris!)
RUN uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"

# Copy source code
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables default
ENV BASE_URL="http://localhost"
ENV PORT=8080
ENV DEBUG=True
ENV DATABASE_URL="postgresql://postgres:Snaillyjuara_1@34.128.50.12:5432/snailly_db?sslmode=require"

# Jalankan aplikasi
CMD ["uv", "run", "python", "server.py"]