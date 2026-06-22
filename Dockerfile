# 1. Gunakan image Python resmi yang ringan sebagai dasar
FROM python:3.10-slim

# 2. Install dependensi sistem operasi linux untuk pemrosesan audio librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Tentukan folder kerja di dalam server cloud
WORKDIR /code

# 4. Salin file requirements.txt ke dalam server
COPY ./requirements.txt /code/requirements.txt

# 5. Install semua library Python yang dibutuhkan model AI kamu
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Salin semua file proyek dari laptopmu ke dalam server cloud
COPY . .

# 7. Jalankan perintah untuk menyalakan server FastAPI via Uvicorn
# Port 7860 digunakan karena merupakan standar port default Hugging Face Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]