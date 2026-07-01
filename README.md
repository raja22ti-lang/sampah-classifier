---
title: Sampah Classifier
emoji: 🗑️
colorFrom: green
colorTo: white
sdk: docker
app_port: 7860
---

# RecycleAI - Web Klasifikasi Sampah dengan Deep Learning (MobileNetV2)

Aplikasi web interaktif premium untuk mendeteksi, mengklasifikasikan jenis sampah secara instan menggunakan model deep learning `garbage_classifier_mobilenetv2.h5` (MobileNetV2), sekaligus membedakan sampah layak daur ulang (*recyclable*) dan tidak layak daur ulang (*non-recyclable*).

## Fitur Utama
*   **Deteksi Otomatis Kategori**: AI mengklasifikasikan sampah ke dalam salah satu dari 10 kategori sampah.
*   **Status Kelayakan Daur Ulang**: Menampilkan badge status secara visual berdasarkan kategori sampah:
    *   **Layak Didaur Ulang (Recyclable)**: Kardus, Pakaian, Kaca, Logam, Kertas, Plastik.
    *   **Tidak Layak Didaur Ulang (Non-Recyclable)**: Sepatu (bahan campuran/komposit), Sampah Residu.
    *   **Dapat Dikomposkan (Organic)**: Sampah Organik (Biological).
    *   **Penanganan Khusus (Hazardous)**: Baterai (Limbah B3).
*   **Desain Modern & Premium**: Menggunakan tema gelap (dark mode), efek glassmorphism, ornamen cahaya (glow orbs), serta animasi pemindaian (scanning) yang dinamis.
*   **Panduan Tindakan**: Dilengkapi dengan deskripsi detail dan petunjuk cara pengelolaan/pemilahan sampah yang tepat.

---

## Struktur Proyek

```text
ARRY_UASComvis/
│
├── garbage_classifier_mobilenetv2.h5   # File model deep learning Anda
├── app.py                              # Backend server Flask
├── inspect_model.py                    # Script untuk memeriksa output kelas model Anda
├── requirements.txt                    # Daftar library python yang diperlukan
├── README.md                           # Petunjuk penggunaan aplikasi (file ini)
│
├── templates/
│   └── index.html                      # Layout Frontend Website
│
└── static/
    ├── css/
    │   └── style.css                   # Gaya & Animasi UI Website (Premium Theme)
    └── js/
        └── main.js                     # Handler input, preview gambar, dan request ke API
```

---

## Langkah Instalasi & Penggunaan

### 1. Prasyarat (Prerequisites)
Pastikan komputer Anda sudah terinstal **Python 3.9 s.d. 3.11**. Anda dapat memverifikasinya melalui terminal:
```bash
python --version
```

### 2. Instalasi Library
Buka Command Prompt (CMD) atau Terminal pada direktori folder proyek ini, kemudian jalankan perintah berikut untuk menginstal semua library pendukung:
```bash
pip install -r requirements.txt
```

### 3. Memeriksa Struktur Model Anda (Opsional)
Jalankan script `inspect_model.py` untuk melihat jumlah kelas asli dari model Anda:
```bash
python inspect_model.py
```
Script ini akan menampilkan ringkasan model (model summary), dimensi input (misal: `224x224`), serta **jumlah kelas output** model Anda (seharusnya menunjukkan `10`).

### 4. Menjalankan Aplikasi Web
Jalankan server Flask dengan perintah berikut:
```bash
python app.py
```
Setelah server berjalan, buka browser Anda dan akses alamat:
**[http://localhost:7860](http://localhost:7860)**

---

## Urutan Kategori pada Model (Alphabetical Order)

Backend `app.py` memetakan index kelas keluaran model secara alfabetis (sesuai dengan default folder generator Keras):
1.  **Indeks 0**: Baterai (Battery) - *Non-Recyclable (B3)*
2.  **Indeks 1**: Sampah Organik (Biological) - *Compostable*
3.  **Indeks 2**: Kardus (Cardboard) - *Recyclable*
4.  **Indeks 3**: Pakaian (Clothes) - *Recyclable*
5.  **Indeks 4**: Kaca (Glass) - *Recyclable*
6.  **Indeks 5**: Logam (Metal) - *Recyclable*
7.  **Indeks 6**: Kertas (Paper) - *Recyclable*
8.  **Indeks 7**: Plastik (Plastic) - *Recyclable*
9.  **Indeks 8**: Sepatu (Shoes) - *Non-Recyclable (Composite)*
10. **Indeks 9**: Sampah Residu (Trash) - *Non-Recyclable*

*Catatan: Jika urutan kelas pada model training Anda berbeda dari urutan alfabetis di atas, Anda dapat menyesuaikan susunan list pada variabel `CLASS_MAPPINGS[10]` di dalam file `app.py`.*

