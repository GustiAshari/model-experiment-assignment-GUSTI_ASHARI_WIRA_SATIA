# AI Model Experiment & Evaluation: Sentiment Analysis on Customer Reviews

## 📌 Deskripsi Assignment
Project ini merupakan eksperimen dan evaluasi terhadap dua pendekatan AI yang berbeda untuk menyelesaikan *use case sentiment analysis* terhadap ulasan pelanggan e-commerce. Proyek ini membandingkan (1) model klasik Machine Learning yang dilatih secara lokal menggunakan Scikit-learn (Logistic Regression), dan (2) Large Language Model (LLM) yang diakses melalui Gemini AI API (`gemini-2.5-flash`) menggunakan pendekatan *prompting*. Kedua pendekatan dijalankan pada data uji (*test set*) yang sama untuk mengukur kelebihan, keterbatasan, dan *trade-off* guna menyusun rekomendasi teknis berbasis data.

---

## 📑 Sumber Dataset
Dataset yang digunakan dalam project ini bersumber dari data ulasan internal tim produk e-commerce:
- **File Input Utama:** `customer_reviews_sentiment.csv`
- **Target Label:** `sentiment` (nilai: `positif` / `negatif`)

---

## 📁 Struktur Folder Project
```text
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   └── model_comparison_summary.png (opsional)
├── README.md
└── requirements.txt
```

---

## 🔍 Kondisi Awal Dataset

Berdasarkan pemeriksaan awal (*Data Inspection*) terhadap dataset `customer_reviews_sentiment.csv`, ditemukan karakteristik dan kondisi awal sebagai berikut:

| Parameter Inspection | Temuan / Detail |
| :--- | :--- |
| **Ukuran Dataset** | 200 baris data |
| **Distribusi Label** | Terdiri dari **110 data positif** dan **90 data negatif** |
| **Metode Evaluasi** | Dipisahkan menggunakan *Stratified Split* (80% Train, 20% Test) dengan `random_state=42` |
| **Ukuran Data Latih (*Train Data*)**| **160 baris data**|
| **Ukuran Data Uji (*Test Data*)**| **40 baris data**|

---

## 🧹 Eksperimen Pendekatan 1 - Model Klasik (Scikit-learn)

Berikut adalah rincian tahapan pemrosesan dan ekstraksi fitur yang diterapkan pada model klasik sesuai dengan urutan kode implementasi:

| Komponen | Metode yang Digunakan | Alasan Pemilihan Metode |
| :--- | :--- | :--- |
| **Text Preprocessing** | `TfidfVectorizer(max_features=1000)` | Mengubah teks ulasan menjadi representasi numerik berbasis bobot frekuensi kata (TF-IDF) untuk menangkap bobot kata kunci penting. |
| **Algoritma Model** | `Logistic Regression(random_state=42)` | Algoritma klasifikasi linier biner yang sangat efisien, cepat, dan bekerja optimal pada teks berdimensi tinggi. |
| **Inference Latency** | Sub-milidetik (< 1 ms) | Proses kalkulasi matriks statistik berjalan sangat instan langsung di *resource* mesin lokal. |
| **Biaya Operasional** | *Zero Variable Cost* (Gratis) | Pemrosesan sepenuhnya lokal tanpa bergantung pada *cloud provider* atau API eksternal. |

---

## ⚙️ Eksperimen Pendekatan 2 - LLM API (Gemini)

Proses klasifikasi teks berbasis *Generative AI* diimplementasikan menggunakan pemanggilan Gemini API sesuai rancangan fungsi di notebook:

1. **Rancangan Teknik Prompting (Zero-Shot):**
   - **Metode:** Menginstruksikan model untuk bertindak sebagai pengklasifikasi sentimen yang kaku. Prompt dirancang khusus agar output model hanya berupa satu kata pilihan (`'positif'` atau `'negatif'`) tanpa tanda baca atau teks penjelasan tambahan.

2. **Pemilihan Parameter Konfigurasi:**
   - **Model:** Menggunakan backend **`gemini-2.5-flash`** melalui SDK `google-genai` terbaru.
   - **`temperature = 0.0`**: Dipilih agar respons LLM bersifat sepenuhnya deterministik, konsisten, kaku, dan tidak imajinatif dalam mendeteksi pola teks.
   - **`max_output_tokens = 5`**: Membatasi konsumsi token agar menekan *latency* jaringan dan biaya penggunaan kuota.

3. **Inference Latency & Rate Limit:**
   - Memerlukan waktu sekitar **1 hingga 1.5 detik per request** karena adanya hambatan jaringan (*network latency*). Diterapkan jeda `time.sleep(1)` di setiap perulangan data untuk menghindari pemblokiran *Rate Limit (RPM)* API.

---

## 📊 Evaluasi dan Perbandingan Performa

Pengujian performa pada **40 baris data uji yang sama** menghasilkan metrik evaluasi sebagai berikut:

### 1. Tabel Perbandingan Hasil Ringkasan

| Metrik Evaluasi | Model Klasik (Logistic Regression + TF-IDF) | LLM API (Gemini 2.5 Flash Zero-Shot) |
| :--- | :---: | :---: |
| **Accuracy** | 0.9000 | **0.9500** |
| **Precision** | 0.8800 | **0.9545** |
| **Recall** | **0.9565** | 0.9545 |
| **F1-Score** | 0.9167 | **0.9545** |

### 2. Confusion Matrix Comparison
*   **Model Klasik:**
    \[\begin{bmatrix} 22 & 1 \\ 3 & 14 \end{bmatrix}\]
*   **LLM API (Gemini):**
    \[\begin{bmatrix} 21 & 2 \\ 0 & 17 \end{bmatrix}\]

### 3. Interpretasi Metrik Ringkas
*   **Accuracy:** Gemini 2.5 Flash mengklasifikasikan 95% data uji dengan benar, sedikit unggul dari Model Klasik yang meraih 90%.
*   **Precision:** Gemini memiliki ketepatan tinggi (95.45%), meminimalkan kesalahan prediksi *false-positive*. Model klasik sering terkecoh oleh teks ulasan negatif yang mengandung kata bermakna ganda.
*   **Recall:** Model Klasik unggul tipis (95.65%), menunjukkan keahliannya menjaring hampir seluruh ulasan positif asli tanpa ada yang terlewat.
*   **F1-Score:** Secara keseimbangan nilai harmonis, LLM Gemini memimpin dengan skor total 95.45%.

---

## ⚠️ Analisis Trade-off & Keterbatasan (Limitation)

### 1. Keterbatasan (Limitation) Model
*   **Model Klasik:** Gagal memahami struktur semantik kontekstual yang bersifat kompleks atau tersirat. Model klasik hanya menghitung frekuensi kata kunci per kata, sehingga tidak bisa membedakan sarkasme atau kalimat penyangkalan.
*   **LLM API:** Sangat bergantung pada keselarasan koneksi jaringan internet. Jika struktur atau format masukan berubah sedikit, terdapat kemungkinan format respons tidak seragam sehingga diperlukan tahap normalisasi *fallback* teks.

### 2. Trade-off Sumber Daya
*   **Waktu Implementasi:** LLM API unggul karena tidak membutuhkan waktu untuk melatih model (*zero training effort*). Model klasik membutuhkan proses penulisan *preprocessing pipeline* teks yang lebih rumit di awal.
*   **Kecepatan & Biaya Produksi:** Model klasik menang mutlak dalam kecepatan respons sub-milidetik dengan biaya operasional Rp0 (gratis). LLM API memakan waktu respons 1+ detik per teks dan memicu biaya operasional komersial berbasis token secara terus-menerus.

---

## 📌 Rekomendasi Technical Approach

Berdasarkan data eksperimen, kami merekomendasikan **Pendekatan Hibrida (Hybrid Approach)** untuk sistem produksi aplikasi e-commerce sesungguhnya dengan pertimbangan komersial:

1.  **Faktor Kenyamanan Pengguna (Latensi):** Fitur otomatisasi ulasan di halaman produk e-commerce membutuhkan respons *real-time*. Kecepatan Model Klasik jauh lebih mendukung skalabilitas sistem web daripada memanggil API eksternal LLM yang memakan waktu 1 detik+ per ulasan.
2.  **Efisiensi Anggaran (Cost Efficiency):** Menghindari pembengkakan biaya pemanggilan token API eksternal untuk jutaan ulasan transaksi pelanggan berskala besar.
3.  **Langkah Tindak Lanjut Teknis:** Manfaatkan LLM Gemini 2.5 Flash di lingkungan pengembangan (*offline*) sebagai alat otomatisasi pelabelan data massal (*data labeling augmentation*). Hasil pelabelan sintetis berkualitas tinggi dari LLM tersebut kemudian digunakan untuk melatih ulang (*retraining*) Model Klasik atau model berbasis *Deep Learning* lokal yang lebih ringan (seperti DistilBERT versi lokal) agar performa akurasinya setara dengan LLM namun dengan kecepatan eksekusi mesin lokal yang instan dan hemat.

---

## 🛠️ Cara Instalasi Dependency

1. Pastikan Anda telah menginstal **Python 3.9+** di komputer lokal Anda.
2. Clone repository ini ke direktori lokal Anda:
   ```bash
   git clone https://github.com
   cd assignment-model-experiment-nama-peserta
   ```
3. Instal seluruh library yang dibutuhkan yang tercatat di `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Cara Menjalankan Notebook

Untuk mereplikasi eksperimen dan melihat hasil tabel evaluasi, jalankan perintah berikut:

1. Atur token API Key Gemini Anda ke dalam *environment variable* sistem operasi:
   - **Linux/macOS:** `export GEMINI_API_KEY="isi_api_key_anda"`
   - **Windows:** `set GEMINI_API_KEY="isi_api_key_anda"`
2. Jalankan teks editor atau Jupyter Notebook:
   ```bash
   jupyter notebook notebook/experiment_notebook.ipynb
   ```
3. Eksekusi semua sel (*Run All Cells*) secara berurutan untuk melihat komparasi metrik secara langsung.
