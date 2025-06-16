# Skin Tone Classifier & Makeup Shade Recommender

Aplikasi web dan desktop untuk klasifikasi warna kulit dan rekomendasi shade make up berbasis pengolahan citra digital manual. Mendukung input gambar (upload) dan kamera realtime (PC/HP, bisa pilih kamera depan/belakang), serta deteksi wajah otomatis.

---

## Fitur Utama
- **Klasifikasi warna kulit** (White, Brown/Tan/Medium, Black) dan rekomendasi shade make up.
- **Pengolahan citra manual**: grayscale, gaussian blur, noise, edge detection, segmentasi kulit.
- **Deteksi wajah otomatis** (face-api.js, crop area wajah untuk hasil maksimal).
- **Input gambar**: upload file atau ambil foto dari kamera (PC/HP, bisa pilih device).
- **Ambil 5 foto otomatis** (1 foto per detik, summary hasil 5 foto).
- **UI modern neo-brutalism** (tebal, warna kontras, border tegas).
- **Offline**: semua library dan model lokal, tanpa CDN.

---

## Instalasi & Menjalankan

1. **Clone/download repo ini** ke komputer kamu.
2. Pastikan struktur folder seperti berikut:
   ```
   /skin-tone-classifier-realtime.html
   /skin-tone-classifier.html
   /face-api.min.js
   /face-api.min.js.map (opsional)
   /weights/
       tiny_face_detector_model-weights_manifest.json
       tiny_face_detector_model-shard1
       ... (file model lain jika diperlukan)
   ```
3. **Buka file HTML** (`skin-tone-classifier-realtime.html` atau `skin-tone-classifier.html`) di browser modern (Chrome, Edge, Firefox).
4. Jika menggunakan fitur kamera, izinkan akses kamera saat diminta.

---

## Penjelasan Fungsi Utama

- **Grayscale**: Konversi manual RGB ke grayscale.
- **Gaussian Blur**: Blur manual menggunakan kernel 3x3.
- **Gaussian Noise**: Menambahkan noise acak ke gambar.
- **Sobel Edge**: Deteksi tepi manual (Sobel operator).
- **Segmentasi Kulit**: Threshold sederhana area kulit pada RGB.
- **Klasifikasi Warna Kulit**: Berdasarkan rata-rata warna area kulit, hasilkan kategori dan rekomendasi shade make up.
- **Deteksi Wajah**: Menggunakan face-api.js, crop otomatis ke area wajah sebelum proses.
- **Ambil 5 Foto Otomatis**: Untuk mode kamera, aplikasi otomatis mengambil 5 foto (1 per detik), lalu menampilkan summary hasil klasifikasi.

---

## Catatan
- Semua proses utama dikerjakan manual di JavaScript/HTML5 Canvas, tanpa library image processing eksternal.
- Model deteksi wajah (weights) harus ada di folder `weights`.
- Untuk penggunaan offline, pastikan semua file JS dan model sudah ada di lokal.

---

## Lisensi
Open source untuk keperluan edukasi dan tugas akhir.

---

## Author
- Prima Hagantha & Team
- Face detection: [face-api.js](https://github.com/justadudewhohacks/face-api.js)

---

# Skin Tone Classifier & Makeup Shade Recommender (Python Streamlit)

Aplikasi Python berbasis Streamlit untuk klasifikasi warna kulit dan rekomendasi shade make up, dengan implementasi manual pengolahan citra digital.

---

## Fitur Utama
- **Klasifikasi warna kulit** (White, Brown/Tan/Medium, Black) dan rekomendasi shade make up.
- **Pengolahan citra manual**: grayscale, konvolusi, gaussian blur, noise, edge detection (Sobel), transformasi Fourier, segmentasi kulit, deblurring (Lucy-Richardson).
- **Input gambar**: upload file atau ambil dari kamera (PC, webcam, IP camera).
- **UI interaktif**: visualisasi hasil setiap tahapan.
- **Penjelasan dan rumus manual** untuk setiap proses.

---

## Instalasi & Menjalankan

1. **Clone/download repo ini** ke komputer kamu.
2. Pastikan sudah terinstall Python 3.8+.
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run pcd.py
   ```
5. Buka browser ke alamat yang tertera (biasanya http://localhost:8501)

---

## Penjelasan Fungsi Utama

- **Grayscale**: Konversi manual RGB ke grayscale.
- **Konvolusi**: Filtering manual (blur, edge, dsb) dengan kernel.
- **Gaussian Blur**: Blur manual menggunakan kernel 3x3.
- **Gaussian Noise**: Menambahkan noise acak ke gambar.
- **Sobel Edge**: Deteksi tepi manual (Sobel operator).
- **Transformasi Fourier**: DFT manual untuk analisis spektrum frekuensi.
- **Segmentasi Kulit**: Threshold manual area kulit pada HSV.
- **Klasifikasi Warna Kulit**: Berdasarkan statistik area kulit, hasilkan kategori dan rekomendasi shade make up.
- **Deblurring**: Lucy-Richardson manual.
- **Input Kamera**: Pilih device kamera, bisa webcam internal/eksternal atau IP camera.

---

## Catatan
- Semua proses utama dikerjakan manual di Python (tanpa fungsi siap pakai OpenCV untuk proses utama).
- Penjelasan dan visualisasi setiap metode tersedia di aplikasi.
- Untuk Windows, deteksi kamera menggunakan modul `wmi`.

---

## Lisensi
Open source untuk keperluan edukasi dan tugas akhir.

---

## Author
- Adaptasi dan pengembangan oleh Prima
- Face detection: OpenCV haarcascade
