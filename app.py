from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import numpy as np
import librosa
import joblib
from pydub import AudioSegment

app = FastAPI(title="TBCheck Backend API")

# Mengizinkan Frontend (HTML) mengakses Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ganti dengan domain frontend saat produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 1. MUAT SEMUA ARTIFACT SEKALI SAAT START (jangan muat per-request)
# =====================================================================
# meta.json -> threshold, urutan kolom, mapping kategori, median, disclaimer
with open("meta.json", "r", encoding="utf-8") as f:
    META = json.load(f)

# config.json -> parameter audio (struktur DATAR, tanpa key "audio")
with open("config.json", "r", encoding="utf-8") as f:
    AUDIO = json.load(f)

FINAL_THRESHOLD = float(META["final_threshold"])     # 0.4
SAMPLE_RATE = AUDIO["sample_rate"]                    # 16000
DURATION = AUDIO["duration"]                          # 0.5
N_MELS = AUDIO["n_mels"]                              # 128
N_MFCC = AUDIO["n_mfcc"]                              # 40
N_FFT = AUDIO["n_fft"]                                # 512
HOP = AUDIO["hop_length"]                             # 128

CLINICAL_COLS = META["clinical_cols"]
NUMERICAL_COLS = META["numerical_cols"]
CATEGORICAL_COLS = META["categorical_cols"]
CATEGORY_MAPPINGS = META["category_mappings"]
MEDIAN_VALUES = META["median_values"]

# Preprocessing objects + model
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

from tensorflow.keras.models import load_model
model = load_model("models/model.keras", compile=False)

# Panjang frame waktu yang model harapkan (dari shape input: 63)
TARGET_FRAMES = 63

print("Model & artifact dimuat. Backend siap. Threshold =", FINAL_THRESHOLD)


# =====================================================================
# 2. KONVERSI AUDIO (apa pun -> WAV 16kHz mono) memakai pydub
# =====================================================================
def preprocess_and_convert_audio(audio_bytes, filename: str):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        fmt = (filename or "").split(".")[-1].lower()
        if fmt == "mp3":
            audio = AudioSegment.from_file(audio_stream, format="mp3")
        elif fmt in ["m4a", "aac"]:
            audio = AudioSegment.from_file(audio_stream, format="m4a")
        elif fmt in ["webm", "ogg"]:
            audio = AudioSegment.from_file(audio_stream)   # butuh ffmpeg
        else:
            audio = AudioSegment.from_file(audio_stream)

        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Gagal memproses format audio {filename}: {str(e)}",
        )


# =====================================================================
# 3. EKSTRAKSI FITUR AUDIO -> bentuk PERSIS seperti yang model minta
#    logmel_input: (1, 63, 128, 1) | mfcc_input: (1, 63, 120)
# =====================================================================
def _fix_frames(feat_2d, target=TARGET_FRAMES):
    """Samakan jumlah frame waktu (sumbu 0) ke `target` dengan pad/truncate."""
    if len(feat_2d) < target:
        pad = np.zeros((target - len(feat_2d), feat_2d.shape[1]))
        feat_2d = np.vstack([feat_2d, pad])
    else:
        feat_2d = feat_2d[:target]
    return feat_2d


def ekstrak_fitur_audio(wav_stream):
    # load + normalisasi + samakan durasi 0.5 detik
    y, sr = librosa.load(wav_stream, sr=SAMPLE_RATE)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # log-mel -> (frames, 128)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP
    )
    logmel = librosa.power_to_db(mel, ref=np.max).T
    logmel = _fix_frames(logmel)                       # (63, 128)

    # mfcc + delta + delta2 -> (frames, 120)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stack = np.vstack([mfcc, d1, d2]).T           # (frames, 120)
    mfcc_stack = _fix_frames(mfcc_stack)               # (63, 120)

    logmel = logmel[np.newaxis, ..., np.newaxis]       # (1, 63, 128, 1)
    mfcc_stack = mfcc_stack[np.newaxis, ...]           # (1, 63, 120)
    return logmel, mfcc_stack


# =====================================================================
# 4. DATA KLINIS -> encode + scale (pakai scaler & encoder dari training)
# =====================================================================
def siapkan_vektor_klinis(age, BMI, reported_cough_dur, sex,
                          weight_loss, smoke_lweek, fever, night_sweats):
    raw = {
        "age": age, "BMI": BMI, "reported_cough_dur": reported_cough_dur,
        "sex": sex, "weight_loss": weight_loss, "smoke_lweek": smoke_lweek,
        "fever": fever, "night_sweats": night_sweats,
    }

    # encode kategorikal pakai mapping (tahan terhadap nilai tak dikenal)
    for col in CATEGORICAL_COLS:
        mapping = CATEGORY_MAPPINGS[col]
        val = str(raw[col])
        if val not in mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Nilai '{val}' untuk '{col}' tidak dikenal. Harus salah satu dari: {list(mapping.keys())}",
            )
        raw[col] = mapping[val]

    # susun sesuai urutan kolom training, lalu scale
    vektor = np.array([[float(raw[c]) for c in CLINICAL_COLS]])   # (1, 8)
    vektor = scaler.transform(vektor)
    return vektor


# =====================================================================
# 5. ENDPOINT SKRINING
# =====================================================================
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True, "threshold": FINAL_THRESHOLD}


@app.post("/api/v1/screening")
async def screening_endpoint(
    age: float = Form(...),
    BMI: float = Form(...),
    reported_cough_dur: float = Form(...),
    sex: str = Form(...),
    weight_loss: str = Form(...),
    smoke_lweek: str = Form(...),
    fever: str = Form(...),
    night_sweats: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    # --- data klinis (sekali, dipakai untuk semua audio) ---
    clinical_vec = siapkan_vektor_klinis(
        age, BMI, reported_cough_dur, sex,
        weight_loss, smoke_lweek, fever, night_sweats
    )

    # --- proses 3 audio, kumpulkan probabilitas ---
    daftar_audio = [audio_1, audio_2, audio_3]
    semua_prob = []
    for af in daftar_audio:
        isi = await af.read()
        wav_stream = preprocess_and_convert_audio(isi, af.filename)
        logmel, mfcc = ekstrak_fitur_audio(wav_stream)
        prob = float(model.predict([logmel, mfcc, clinical_vec], verbose=0).flatten()[0])
        semua_prob.append(prob)

    # rata-rata 3 rekaman lebih stabil daripada nilai maksimum
    skor_final = float(np.mean(semua_prob))
    status = "positif" if skor_final >= FINAL_THRESHOLD else "negatif"

    return {
        "status": status,
        "skor_final": round(skor_final, 4),          # untuk log internal
        "detail_skor_batuk": [round(s, 4) for s in semua_prob],
        "threshold": FINAL_THRESHOLD,
        "disclaimer": META["disclaimer"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
