# Approach - BDC Internal 2026 Case 1 (SatriaData)

Repositori ini memuat pendekatan pemodelan *End-to-End* untuk memecahkan masalah klasifikasi *multiclass* (8 kelas) pada domain *User Generated Content* (UGC) Twitter berbahasa Indonesia mengenai program Makan Bergizi Gratis (MBG). 

Metodologi ini dirancang eksklusif dengan pola pikir **Kaggle Master**, di mana prioritas tertinggi diletakkan pada penciptaan *Generalization Shield* (perisai anti-overfitting) yang kokoh, mengingat kompetisi ini adalah *Blind Test* tanpa adanya *Public Leaderboard*.

---

## 1. Pipeline & Arsitektur Utama
- **Model Backbone**: XLM-RoBERTa-Large (`xlm-roberta-large`)
- **Framework**: PyTorch + HuggingFace Transformers
- **Strategi Evaluasi**: 5-Fold Stratified K-Fold Cross-Validation (menjaga distribusi kelas minoritas di setiap *fold*).
- **Teknik Regularisasi Ekstrem**: 
  - *Multi-Sample Dropout*: Menggunakan 5 *head* secara paralel (dropout rate 0.2) untuk menstabilkan model dengan parameter masif (550M parameter).
  - *Class-Weighted CrossEntropyLoss*: Memberikan penalti matematis berdasarkan invers frekuensi kelas untuk mengatasi ketidakseimbangan data yang ekstrem (seperti pada kelas "Ekonomi" yang hanya memiliki prevalensi 2.9%).
  - *Adversarial Training (FGM)*: Menyuntikkan *noise* mikroskopis ke dalam ruang *embedding* kata (`epsilon=0.5`) saat pelatihan agar model tidak sekadar "menghafal" data *Train*.
- **Optimasi Lingkungan**: *Automatic Mixed Precision (AMP)* dan *Gradient Accumulation* digunakan untuk mengeksekusi *batch* besar tanpa melebihi batas memori GPU.

---

## 2. Studi Ablasi Pemodelan (Model Ablation)
Perjalanan kami mencari batas representasi teks dilakukan melalui puluhan iterasi eksperimen yang terukur. XLM-RoBERTa-Large bukanlah tebakan instan, melainkan puncak piramida dari rentetan uji coba berikut:

1. **Classical ML (TF-IDF Hybrid + LinearSVC): 0.5862**
   - Batas absolut (*ceiling*) untuk pendekatan *Machine Learning* Klasik. Meskipun sangat efisien dan kebal terhadap salah ketik (*typo*), representasi leksikal/frekuensi gagal mengenali konteks lintas-kalimat (gagal membedakan irisan tipis antara topik "Anggaran" dan "Tata Kelola").
2. **Dense Statis (Word2Vec / FastText): ~0.2399 - 0.4356**
   - Hasil terburuk. Data latih sebesar 5.000 sampel terbukti jauh dari cukup untuk melatih matriks ruang semantik *dense* secara mandiri.
3. **IndoBERT-Large-p1 (The Deep Learning Transition): 0.6221**
   - Keputusan bertransisi ke model berbasis *Transformer Pre-trained* langsung menembus skor 60%. IndoBERT mapan secara gramatikal, namun mulai kewalahan membedah ambiguitas subkelas pada data yang sangat tidak seimbang (*imbalanced*).
4. **MDeBERTa-V3-Base: 0.1250 (Gagal Total / Mode Collapse)**
   - Kami menguji arsitektur DeBERTa yang terkenal kuat, namun model ini terbukti sangat tidak stabil pada dataset ini. Di epoch pertama, DeBERTa mengalami *catastrophic forgetting* dan berakhir memprediksi secara buta hanya pada satu kelas mayoritas (1/8 probabilitas atau 12.5%).
5. **XLM-RoBERTa-Large + FGM: ~0.6808 (Mahakarya Final)**
   - Kami mengerahkan arsitektur raksasa (550 Juta Parameter). Dengan injeksi *Adversarial Training (FGM)*, model dipaksa untuk benar-benar memahami makna, bukan menghafal *keyword*. Hasilnya adalah lonjakan akurasi yang masif hingga mendekati 70% di beberapa *fold*.

---

## 3. Eksperimen Preprocessing: "Surgical" vs "Destructive"
Dalam kompetisi ini, kami menemukan **The NLP Paradox**: asumsi standar bahwa teks sekotor Twitter harus "dibersihkan secara agresif" terbukti **salah secara empiris** saat dihadapkan pada model *Deep Learning*. Berikut evolusinya:

1. **Fase Mentah (Naked Text): Skor 0.6221**
   - Saat teks dibiarkan 100% kotor (URL, Mention `@`, dan Emoji utuh), IndoBERT justru mencetak rekor tingginya. Model menjadikan elemen-elemen ini sebagai *attention clues*.
2. **Fase Destructive Cleaning: Skor 0.6098 (TURUN)**
   - Kami menghapus URL, Mention, dan *Stop words*. Hasilnya, skor merosot tajam. Analisis Data Eksploratif (EDA) membuktikan bahwa 59% *tweet* memiliki *Mention* (`@`) yang berkorelasi absolut dengan kelas tertentu (misalnya `@prabowo` selalu merujuk pada "Politik" atau "Pemerintah"). Menghapusnya membuat model buta huruf terhadap subjek krusial.
3. **Fase Semantic Normalization: Skor 0.6175 (TURUN)**
   - Kami mencoba abstraksi logis (URL diubah menjadi teks `"tautan web"`, *mention* diubah menjadi `"akun netizen"`). *Transformer* menolak keras rasionalisasi manusia ini karena membuang variansi linguistik alami pada manifold semantik.
4. **Kesimpulan Final (Surgical Preprocessing): Melesat ke ~0.6808**
   - Teks **wajib** dibiarkan senatural mungkin. Intervensi hanya dilakukan pada tingkat mikro untuk mencegah *Out-of-Vocabulary* (OOV), bukan untuk membuang konteks. Kombinasi final kami:
     - **Pertahankan** URL, Mentions, dan Emoji.
     - **Obfuscation Reduction**: Menciutkan karakter berlebih (`basiii` menjadi `basi`).
     - **Targeted Leetspeak**: Menerjemahkan angka kamuflase menjadi abjad (`p3merintah` menjadi `pemerintah`, `4nggaran` menjadi `anggaran`).
     - **Kamus Slang Terbatas**: Normalisasi kata gaul yang dominan di domain MBG.
     - **Deduplikasi Ekstrem**: Membuang 100% *tweet buzzer/spam* yang teksnya berulang pada data *Train* untuk mencegah model terilusi oleh *majority class bias*.

---

## 4. Struktur File
```text
├── ROBERTAterbaik_documented.ipynb   # Source code utama (End-to-End Pipeline)
├── requirements.txt                  # Daftar pustaka Python yang dibutuhkan
├── InginSepertiPalsnups.xlsx           # File prediksi akhir siap unggah
└── README.md                         # Panduan reproduksi (file ini)
```

## 5. Cara Menjalankan Kode (Reproduktibilitas)
Skrip ini telah dirancang 100% *End-to-End*. Demi kestabilan VRAM dan kecepatan memori, arsitektur *Automatic Mixed Precision (AMP)* mengharuskan Anda mengeksekusi ini di Kaggle atau PC dengan GPU minimal 16GB.

### Tahap 1: Persiapan Lingkungan
1. **Accelerator (GPU):** Set *Hardware* ke **GPU T4 x2** (atau P100 jika tersedia).
2. **Internet Access:** Pastikan *toggle* **Internet ON**. (*Krusial untuk mengunduh arsitektur dan bobot pre-trained `xlm-roberta-large` sebesar 2.2GB dari HuggingFace*).
3. **Language:** Python 3.

### Tahap 2: Manajemen Dataset
Unggah data mentah kompetisi sebagai *Kaggle Dataset* (Pilih `Add Data` -> `Upload`):
- `case_1_labeled_data.xlsx - Sheet1.csv`
- `case_1_text_to_predict.xlsx - Sheet1.csv`
- `case_1_template_sheet.xlsx - Sheet1.csv`

### Tahap 3: Konfigurasi Path di Skrip Utama
Buka file skrip/notebook utama (`ROBERTAterbaik_documented.ipynb`). Di baris konfigurasi teratas, pastikan *path* mengarah tepat ke lokasi direktori data yang baru saja Anda unggah:
```python
TRAIN_PATH = "/kaggle/input/NAMA_DATASET_ANDA/case_1_labeled_data.xlsx - Sheet1.csv"
TEST_PATH = "/kaggle/input/NAMA_DATASET_ANDA/case_1_text_to_predict.xlsx - Sheet1.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/NAMA_DATASET_ANDA/case_1_template_sheet.xlsx - Sheet1.csv"
```

### Tahap 4: Eksekusi Paripurna (*Run All*)
1. Klik **Run All**.
2. **Apa yang terjadi di balik layar?** Skrip akan mengeksekusi *Surgical Preprocessing* (Deduplikasi, Leetspeak, dll), melakukan pembelahan 5-Fold Stratified K-Fold secara otomatis, dan melatih model raksasa XLM-RoBERTa secara utuh.
3. **Soft-Voting Antar-Fold:** Probabilitas prediksi dari kelima lipatan (Fold 1 hingga Fold 5) model RoBERTa akan dirata-ratakan secara matematis. Ini menjamin prediksi akhir yang sangat stabil.
4. **Estimasi Waktu:** Keseluruhan iterasi memakan waktu sekitar **~35 hingga 40 menit**.
5. **Output Target:** Sistem akan membuang hasil inferensi secara otomatis ke dalam `/kaggle/working/submission_roberta.xlsx`. File inilah lembar jawaban mutlak Anda untuk di-*submit* ke *Leaderboard* BDC.
