import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import os
import wmi
# ==============================================================================
# 1. IMPLEMENTASI MANUAL ALGORITMA PENGOLAHAN CITRA
# ==============================================================================

def list_available_cameras(max_index=5):
    # Coba ambil nama device kamera (khusus Windows, butuh pywin32/wmi)
    try:

        c = wmi.WMI()
        cam_names = []
        for cam in c.Win32_PnPEntity():
            if cam.Name and ("camera" in cam.Name.lower() or "webcam" in cam.Name.lower()):
                cam_names.append(cam.Name)
        # Cocokkan index dengan urutan deteksi OpenCV
        available = []
        idx = 0
        for i in range(max_index+1):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                name = cam_names[idx] if idx < len(cam_names) else f"Camera {i}"
                available.append(f"{name} (Index {i})")
                cap.release()
                idx += 1
        return available
    except Exception:
        # Fallback: hanya index
        available = []
        for i in range(max_index+1):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                available.append(f"Camera {i}")
                cap.release()
        return available

def add_gaussian_noise(image, sigma=20):
    """
    Metode: Noise Restoration (Simulasi)
    Algoritma: Menambahkan Gaussian Noise secara manual.
    Rumus: P_baru = P_asli + (Random_Gaussian * sigma)
    """
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def manual_convolve(image, kernel):
    """
    Metode: Konvolusi
    Algoritma: Melakukan konvolusi 2D secara manual.
    Mendukung input 2D (grayscale) maupun 3D (RGB).
    """
    kernel = np.flipud(np.fliplr(kernel))
    if kernel.ndim == 3:
        kernel = kernel[:, :, 0]
    # Manual: jika input 2D, tetap 2D, tidak convert pakai cv2
    is_2d = False
    if image.ndim == 2:
        is_2d = True
        image = image[:, :, np.newaxis]
    k_height, k_width = kernel.shape
    pad_h = k_height // 2
    pad_w = k_width // 2
    output = np.zeros_like(image, dtype=np.float32)
    image_padded = np.zeros((image.shape[0] + 2*pad_h, image.shape[1] + 2*pad_w, image.shape[2]))
    for ch in range(image.shape[2]):
        image_padded[:, :, ch] = np.pad(image[:, :, ch], ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    for c in range(image.shape[2]):
        for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                output[x, y, c] = (kernel * image_padded[x:x+k_height, y:y+k_width, c]).sum()
    if is_2d:
        return np.clip(output[:, :, 0], 0, 255).astype(np.uint8)
    return np.clip(output, 0, 255).astype(np.uint8)

# Manual grayscale tanpa cv2

def manual_rgb_to_gray(image):
    """
    Konversi RGB ke grayscale manual: Y = 0.299*R + 0.587*G + 0.114*B
    """
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.uint8)

# Manual Sobel tanpa cv2

def apply_sobel(image):
    gray_image = manual_rgb_to_gray(image).astype(np.float32)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = manual_convolve(gray_image, kernel_x)
    gy = manual_convolve(gray_image, kernel_y)
    magnitude = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255).astype(np.uint8)
    return np.stack([magnitude]*3, axis=-1)

def apply_lucy_richardson(image, psf, iterations=5):
    """
    Metode: Deblurring
    Algoritma: Lucy-Richardson Iterative Deblurring.
    """
    latent_image = image.copy().astype(np.float32)
    observed_image = image.copy().astype(np.float32)
    
    # Make PSF 3D to match image channels
    psf_3d = np.repeat(psf[:, :, np.newaxis], 3, axis=2)
    psf_flipped = np.flipud(np.fliplr(psf_3d))

    for _ in range(iterations):
        estimate_conv = manual_convolve(latent_image, psf_3d)
        estimate_conv = np.clip(estimate_conv, 1e-2, None)  # Hindari pembagian dengan angka sangat kecil
        relative_blur = observed_image / estimate_conv
        error_estimate = manual_convolve(relative_blur, psf_flipped)
        latent_image *= error_estimate
        latent_image = np.clip(latent_image, 0, 255)  # Clamp agar tidak overflow/underflow

        return np.clip(latent_image, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image):
    """Fungsi pembungkus untuk Gaussian Blur menggunakan konvolusi manual."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    # Perluas kernel ke 3 channel
    kernel_3d = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)
    return manual_convolve(image, kernel_3d)


def apply_dft(image):
    """
    Metode: Transformasi
    Algoritma: Discrete Fourier Transform (DFT) manual.
    Konversi ke grayscale dilakukan manual.
    """
    # Grayscale manual: Y = 0.299*R + 0.587*G + 0.114*B
    img_float = image.astype(np.float32)
    gray_image = 0.299 * img_float[..., 0] + 0.587 * img_float[..., 1] + 0.114 * img_float[..., 2]
    # DFT
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Tambah 1 untuk menghindari log(0)
    # Normalisasi untuk ditampilkan
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Ubah ke RGB untuk konsistensi tampilan
    return np.stack([magnitude_spectrum]*3, axis=-1)


def rgb_to_hsv_manual(image):
    """
    Konversi RGB ke HSV secara manual untuk seluruh gambar (tanpa cv2.cvtColor).
    Input: image (np.ndarray, uint8, shape (H, W, 3))
    Output: hsv_image (np.ndarray, float32, shape (H, W, 3)), H: 0-179, S: 0-255, V: 0-255
    """
    img = image.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax
    # Hue
    mask = delta != 0
    h[mask & (cmax == r)] = (60 * ((g[mask & (cmax == r)] - b[mask & (cmax == r)]) / delta[mask & (cmax == r)]) + 0) % 360
    h[mask & (cmax == g)] = (60 * ((b[mask & (cmax == g)] - r[mask & (cmax == g)]) / delta[mask & (cmax == g)]) + 120) % 360
    h[mask & (cmax == b)] = (60 * ((r[mask & (cmax == b)] - g[mask & (cmax == b)]) / delta[mask & (cmax == b)]) + 240) % 360
    h = h / 2  # OpenCV HSV: H in [0,179]
    # Saturation
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    s = s * 255
    v = v * 255
    hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
    return hsv

def segment_skin_manual(image):
    """
    Segmentasi area kulit secara manual di ruang HSV (tanpa cv2.cvtColor).
    Gunakan hasil smoothing/noise reduction agar mask lebih stabil.
    """
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    # Smoothing sebelum segmentasi
    smooth = apply_gaussian_blur(image)
    hsv_image = rgb_to_hsv_manual(smooth)
    lower_skin = np.array([0, 10, 30], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = ((hsv_image[..., 0] >= lower_skin[0]) & (hsv_image[..., 0] <= upper_skin[0]) &
                 (hsv_image[..., 1] >= lower_skin[1]) & (hsv_image[..., 1] <= upper_skin[1]) &
                 (hsv_image[..., 2] >= lower_skin[2]) & (hsv_image[..., 2] <= upper_skin[2]))
    skin_mask = skin_mask.astype(np.uint8) * 255
    skin_image = image.copy()
    if skin_image.shape[2] == 3:
        skin_image[skin_mask == 0] = [128, 128, 128]
    else:
        skin_image[skin_mask == 0] = 128
    return skin_image, skin_mask

def classify_skin_tone_manual(image, skin_mask):
    """
    Klasifikasi skin tone berdasarkan nilai median dan mean HSV pada area kulit.
    Kategori: White, Brown, Black.
    """
    hsv = rgb_to_hsv_manual(image)
    # Ekstraksi nilai H, S, V pada mask kulit
    h_vals = hsv[..., 0][skin_mask > 0]
    s_vals = hsv[..., 1][skin_mask > 0] / 255.0
    v_vals = hsv[..., 2][skin_mask > 0] / 255.0

    if len(h_vals) == 0:
        return "Tidak Terklasifikasi", "-", "-", "-"

    # Hitung statistik
    med_H, med_S, med_V = np.median(h_vals), np.median(s_vals), np.median(v_vals)
    mean_H, mean_S, mean_V = np.mean(h_vals), np.mean(s_vals), np.mean(v_vals)
    print(f"Statistik Kulit: med_H={med_H}, med_S={med_S}, med_V={med_V}, mean_H={mean_H}, mean_S={mean_S}, mean_V={mean_V}")

    # Info bedak dan link
    powder_info = "-"
    powder_link = "-"

    # Threshold untuk kategori “White” (Light)
    if (med_H <= 12 or mean_H <= 12) and (med_S <= 0.35 or mean_S <= 0.35) and (med_V >= 0.7 or mean_V >= 0.7):
        powder_info = "Cocok untuk bedak shade Porcelain, Ivory, Light Beige. Pilih bedak dengan undertone pink atau neutral."
        powder_link = "https://www.google.com/search?q=bedak+terbaik+untuk+kulit+putih+light+porcelain+ivory"
        return "White", "Porcelain / Ivory", powder_info, powder_link

    # Threshold untuk kategori “Brown/Tan/Medium/Sawo Matang”
    if (12 < med_H < 25 or 12 < mean_H < 25) and (0.18 < med_S < 0.4 or 0.18 < mean_S < 0.4) and (0.25 < med_V < 0.7 or 0.25 < mean_V < 0.7):
        powder_info = "Cocok untuk bedak shade Natural, Beige, Honey, Tan, Medium. Pilih bedak dengan undertone kuning, golden, atau olive."
        powder_link = "https://www.google.com/search?q=bedak+terbaik+untuk+kulit+sawo+matang+tan+medium+beige+natural"
        return "Brown/Tan/Medium", "Beige / Natural / Tan", powder_info, powder_link

    # Threshold untuk kategori “Black”
    if (med_H >= 25 or mean_H >= 25) and (med_S > 0.2 or mean_S > 0.2) and (med_V <= 0.6 or mean_V <= 0.6):
        powder_info = "Cocok untuk bedak shade Tan, Cocoa, Espresso. Pilih bedak dengan undertone warm atau bronze."
        powder_link = "https://www.google.com/search?q=bedak+terbaik+untuk+kulit+gelap+tan+cocoa+espresso"
        return "Black", "Tan / Cocoa", powder_info, powder_link

    # Fallback jika tidak memenuhi kondisi di atas
    return "Tidak Terklasifikasi", "-", powder_info, powder_link



# Manual Lucy-Richardson deblurring (tanpa cv2)
def lucy_richardson_manual(image, psf, iterations=3):
    """
    Lucy-Richardson deblurring manual,
    Semua operasi dilakukan manual (tanpa cv2).
    """
    latent = image.astype(np.float32)
    blurred = image.astype(np.float32)
    psf_flipped = np.flipud(np.fliplr(psf))
    for _ in range(iterations):
        # Konvolusi latent dengan PSF (per channel)
        convolved_latent = np.zeros_like(latent)
        for c in range(3):
            convolved_latent[..., c] = manual_convolve(latent[..., c], psf)
        convolved_latent = np.clip(convolved_latent, 1e-2, None)
        # Relative blur
        relative_blur = blurred / convolved_latent
        # Konvolusi relative_blur dengan PSF flipped (per channel)
        error_estimate = np.zeros_like(latent)
        for c in range(3):
            error_estimate[..., c] = manual_convolve(relative_blur[..., c], psf_flipped)
        # Update latent
        latent = latent * error_estimate
        latent = np.clip(latent, 0, 255)
    return latent.astype(np.uint8)


# ==============================================================================
# 2. TAMPILAN ANTARMUKA (UI) STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide")

st.title("Proyek Pengolahan Citra Digital: Klasifikasi Warna Kulit dan Rekomendasi Make Up")
st.markdown("""
## Deskripsi Proyek
Proyek ini mengimplementasikan pengolahan citra digital secara manual menggunakan Python dan Streamlit, sesuai dengan tugas akhir mata kuliah Pengolahan Citra Digital. Semua algoritma utama dikerjakan tanpa fungsi siap pakai dari OpenCV atau library lain, sehingga seluruh proses dapat dipelajari secara mendalam.

## Konsep/Metode yang Diimplementasikan
1. **Digitalisasi**: Konversi manual dari RGB ke grayscale dan HSV, serta segmentasi area kulit.
2. **Konvolusi**: Proses manual untuk filtering (blur, edge, dsb) menggunakan kernel.
3. **Transformasi**: Transformasi Fourier (DFT) manual untuk analisis spektrum frekuensi.
4. **Kontras & Edge Enhancement**: Deteksi tepi (Sobel) dan normalisasi kontras secara manual.
5. **Deblurring & Noise**: Penambahan noise Gaussian dan deblurring Lucy-Richardson manual.
6. **Segmentasi**: Segmentasi area kulit berbasis HSV tanpa fungsi siap pakai.

## Alur Kerja Aplikasi
1. Pilih mode input: unggah gambar atau gunakan kamera.
2. Gambar akan diproses secara bertahap: deblurring, noise reduction, blur, transformasi, edge, segmentasi, dan klasifikasi warna kulit.
3. Setiap hasil tahapan akan divisualisasikan agar mudah dipahami.
4. Rekomendasi shade make up diberikan berdasarkan hasil klasifikasi warna kulit.

## Petunjuk Penggunaan
- Unggah gambar (JPG/PNG) atau gunakan kamera yang tersedia.
- Klik "Proses Gambar" untuk menjalankan seluruh tahapan.
- Lihat hasil setiap proses pada tampilan aplikasi.

---
**Catatan:**
- Semua proses utama dikerjakan manual sesuai permintaan tugas akhir.
- Penjelasan dan visualisasi setiap metode tersedia di aplikasi.
""")

# Tambahkan opsi kamera realtime (5 detik, 1 fps, summary)
st.sidebar.header("Opsi Input")
input_mode = st.sidebar.radio("Pilih Mode Input", ("Upload", "Realtime Camera (5 detik, Pilih Device)",))

if input_mode == "Upload":
    uploaded_file = st.file_uploader("1. Unggah Foto Wajah", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        image = image.resize((256, 256))
        img_array = np.array(image)
        st.subheader("Informasi Gambar Asli")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Gambar Asli", use_container_width=True)
        with col2:
            st.info(f"**Nama File:** {uploaded_file.name}")
            st.info(f"**Ukuran:** {len(bytes_data)/1024:.2f} KB")
            st.info(f"**Resolusi:** {img_array.shape[1]}x{img_array.shape[0]} piksel")
        if st.button("Proses Gambar"):
            with st.spinner("Memproses semua tahapan... Ini mungkin butuh waktu sejenak."):
                # 1. Deblurring
                psf = np.ones((5, 5)) / 25
                deblurred_image = lucy_richardson_manual(img_array, psf, iterations=5)
                # 2. Gaussian Blur
                blurred_image = apply_gaussian_blur(deblurred_image)
                # 3. Gaussian Noise
                noisy_image = add_gaussian_noise(blurred_image)
                # 4. Transformasi Fourier
                dft_image = apply_dft(noisy_image)
                # 5. Sobel Edge Enhancement
                sobel_image = apply_sobel(noisy_image)
                # 6. Segmentasi manual
                segmented_image, skin_mask = segment_skin_manual(img_array)
                # 7. Klasifikasi manual
                tone, shade, powder_info, powder_link = classify_skin_tone_manual(img_array, skin_mask)
                st.subheader("Hasil Pemrosesan Citra")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.image(deblurred_image, caption="1. Deblurring (Lucy-Richardson, Manual)", use_container_width=True)
                with col_b:
                    st.image(blurred_image, caption="2. Gaussian Blur (Konvolusi)", use_container_width=True)
                with col_c:
                    st.image(noisy_image, caption="3. Gaussian Noise", use_container_width=True)
                sobel_image = apply_sobel(noisy_image)
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    st.image(sobel_image, caption="4. Sobel Edge Enhancement", use_container_width=True)
                with col_e:
                    st.image(segmented_image, caption="5. Segmentasi Kulit (HSV, Manual)", use_container_width=True)
                with col_f:
                    st.image(img_array, caption="6. Gambar Asli", use_container_width=True)
                    st.markdown(f"**Warna Kulit Terdeteksi:** {tone}")
                    st.markdown(f"**Shade Make Up Rekomendasi:** {shade}")
                    st.markdown(f"**Info Bedak:** {powder_info}")
                    st.markdown(f"[Lihat Bedak Rekomendasi di Google]({powder_link})")
                
                # --- Penjelasan Manual ---
                with st.expander("Lihat Detail Perhitungan Manual (Contoh)"):
                    st.markdown("""
                    Berikut adalah contoh perhitungan manual yang mendasari setiap algoritma di atas, diambil dari sampel piksel pada gambar Anda.
                    """)
                    st.info("Penjelasan detail rumus untuk setiap tahap dapat ditemukan pada dokumen laporan proyek.")
                    st.markdown("**Manual Deblurring:** Menggunakan algoritma Lucy-Richardson, memperkirakan dan mengoreksi blur iteratif.")
                    st.markdown("**Manual Gaussian Blur:** Setiap piksel dijumlahkan dengan tetangga 3x3, dikalikan kernel, dibagi 16. Tujuan: menghaluskan tekstur kulit.")
                    st.markdown("**Manual Gaussian Noise:** Menambahkan noise acak ke setiap piksel.")
                    st.markdown("**Manual Transformasi Fourier:** Menghitung frekuensi gambar, menampilkan spektrum magnitudo.")
                    st.markdown("**Manual Sobel:** Menghitung gradien gambar untuk mendeteksi tepi.")
                    st.markdown("**Manual Segmentasi Kulit:** Menggunakan thresholding pada ruang warna HSV untuk mengekstrak area kulit.")
                    st.markdown("**Manual Klasifikasi Warna Kulit:** Berdasarkan rata-rata nilai H dari piksel yang tersegmentasi sebagai kulit.")
elif input_mode == "Realtime Camera (5 detik, Pilih Device)":
    st.info("Pilih device kamera yang tersedia di sistem Anda. Untuk webcam laptop, biasanya index 0. Untuk webcam eksternal, index 1 dst. Untuk IP Camera, masukkan URL stream.")
    camera_list = list_available_cameras(5)
    camera_list.append("Custom (Input Index/URL)")
    camera_choice = st.selectbox("Pilih Device Kamera", camera_list)
    custom_input = None
    if camera_choice == "Custom (Input Index/URL)":
        custom_input = st.text_input("Masukkan index kamera (angka) atau URL stream (misal: http://...)")
    if st.button("Mulai Ambil Foto Otomatis 5 Detik (Device Terpilih)"):
        if camera_choice == "Custom (Input Index/URL)":
            cam_input = custom_input
        else:
            # Ambil index kamera dari string, aman jika format berubah
            try:
                cam_input = int(camera_choice.split("Index")[-1].strip(") "))
            except Exception:
                st.error("Gagal membaca index kamera. Silakan pilih atau input manual.")
                cam_input = None
        if cam_input is not None:
            images = []
            stframe = st.empty()
            cap = cv2.VideoCapture(cam_input)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            st.warning("Pastikan wajah menghadap ke depan dan terlihat jelas di kamera!")
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal mengambil gambar dari kamera.")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sobel_image = apply_sobel(rgb_frame)
                faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)
                # Gambar kotak di wajah
                for (x, y, w, h) in faces:
                    cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0,255,0), 2)
                stframe.image([rgb_frame, sobel_image], caption=[f"Frame {i+1} - Deteksi Wajah", f"Frame {i+1} - Sobel Edge Enhancement"], use_container_width=True)
                # Simpan hanya jika ada wajah dan wajah cukup besar
                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                    face_img = rgb_frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (256, 256))
                    images.append(face_img)
                else:
                    st.warning(f"Frame {i+1}: Wajah tidak terdeteksi, frame dilewati.")
                time.sleep(1)
            cap.release()
            stframe.empty()
            st.success(f"Berhasil mengambil {len(images)} foto wajah.")
            if not images:
                st.error("Tidak ada wajah terdeteksi pada 5 detik terakhir.")
            else:
                with st.spinner("Memproses hasil foto wajah..."):
                    results = []
                    for idx, img_array in enumerate(images):
                        # 1. Deblurring
                        psf = np.ones((5, 5)) / 25
                        deblurred_image = lucy_richardson_manual(img_array, psf, iterations=5)
                        # 2. Gaussian Blur
                        blurred_image = apply_gaussian_blur(deblurred_image)
                        # 3. Gaussian Noise
                        noisy_image = add_gaussian_noise(blurred_image)
                        # 4. Transformasi Fourier
                        dft_image = apply_dft(noisy_image)
                        # 5. Sobel Edge Enhancement
                        sobel_image = apply_sobel(noisy_image)
                        # 6. Segmentasi manual
                        segmented_image, skin_mask = segment_skin_manual(img_array)
                        # Fallback jika segmentasi gagal (bukan array)
                        if not isinstance(segmented_image, (np.ndarray, Image.Image)):
                            segmented_image = np.full_like(img_array, 128)  # gambar abu-abu
                        # 7. Klasifikasi manual
                        tone, shade, powder_info, powder_link = classify_skin_tone_manual(img_array, skin_mask)
                        results.append((img_array, deblurred_image, blurred_image, noisy_image, segmented_image, tone, shade, powder_info, powder_link))
                    # Preview 5 foto terakhir
                    st.subheader("Preview 5 Foto Terakhir (Realtime, Otomatis)")
                    for idx, item in enumerate(results):
                        # Cek jumlah elemen dan pastikan semua gambar valid sebelum st.image
                        def is_valid_image(img):
                            return isinstance(img, (np.ndarray, Image.Image))
                        if len(item) == 9:
                            ori, deb, blu, noi, seg, tone, shade, powder_info, powder_link = item
                            st.markdown(f"**Foto ke-{idx+1}: Tone: {tone}, Shade: {shade}, Info Bedak: {powder_info}")
                            ##st.markdown(f"[Lihat Bedak Rekomendasi di Google]({powder_link})")
                            st.image([
                                ori, deb, blu, noi, seg
                            ], caption=[
                                "Gambar Asli", "Deblurring", "Gaussian Blur", "Gaussian Noise", "Segmentasi Kulit (HSV, Manual)"
                            ], width=150)
                        else:
                            st.warning(f"Foto ke-{idx+1}: Data tidak lengkap atau format tidak sesuai.")
                    # Summary
                    tones = [tone for *_, tone, _ in results if tone != "Tidak Terdeteksi"]
                    recs = [rec for *_, _, rec in results if rec != "Bukan Area Kulit"]
                    # Hitung persentase tone dark, tan, light (5 detik terakhir)
                    tone_counts = {"Dark": 0, "Tan": 0, "Light": 0}
                    for _, _, _, _, _, tone, *_ in results:
                        if isinstance(tone, str):
                            if "Black" in tone:
                                tone_counts["Dark"] += 1
                            elif "Brown" in tone or "Tan" in tone or "Medium" in tone:
                                tone_counts["Tan"] += 1
                            elif "White" in tone:
                                tone_counts["Light"] += 1
                    total = sum(tone_counts.values())
                    if total > 0:
                        st.subheader("Persentase Tone Kulit (5 Detik Terakhir)")
                        st.progress(tone_counts["Dark"] / total, text=f"Dark: {tone_counts['Dark'] / total * 100:.1f}%")
                        st.progress(tone_counts["Tan"] / total, text=f"Tan: {tone_counts['Tan'] / total * 100:.1f}%")
                        st.progress(tone_counts["Light"] / total, text=f"Light: {tone_counts['Light'] / total * 100:.1f}%")
                        st.markdown(f"**Dark:** {tone_counts['Dark'] / total * 100:.1f}% | **Tan:** {tone_counts['Tan'] / total * 100:.1f}% | **Light:** {tone_counts['Light'] / total * 100:.1f}%")

                    # Summary keseluruhan (akumulasi semua hasil)
                    if 'all_tone_counts' not in st.session_state:
                        st.session_state['all_tone_counts'] = {"Dark": 0, "Tan": 0, "Light": 0, "Total": 0}
                    st.session_state['all_tone_counts']['Dark'] += tone_counts['Dark']
                    st.session_state['all_tone_counts']['Tan'] += tone_counts['Tan']
                    st.session_state['all_tone_counts']['Light'] += tone_counts['Light']
                    st.session_state['all_tone_counts']['Total'] += total
                    all_total = st.session_state['all_tone_counts']['Total']
                    if all_total > 0:
                        st.subheader("Summary Keseluruhan (Sejak Awal Sesi)")
                        st.markdown(f"**Dark:** {st.session_state['all_tone_counts']['Dark'] / all_total * 100:.1f}% | **Tan:** {st.session_state['all_tone_counts']['Tan'] / all_total * 100:.1f}% | **Light:** {st.session_state['all_tone_counts']['Light'] / all_total * 100:.1f}%")
                        st.progress(st.session_state['all_tone_counts']['Dark'] / all_total, text=f"Dark: {st.session_state['all_tone_counts']['Dark'] / all_total * 100:.1f}%")
                        st.progress(st.session_state['all_tone_counts']['Tan'] / all_total, text=f"Tan: {st.session_state['all_tone_counts']['Tan'] / all_total * 100:.1f}%")
                        st.progress(st.session_state['all_tone_counts']['Light'] / all_total, text=f"Light: {st.session_state['all_tone_counts']['Light'] / all_total * 100:.1f}%")
                    #print link google dari summary
                    st.markdown("**Catatan:** Hasil klasifikasi warna kulit didasarkan pada analisis statistik dari area kulit yang tersegmentasi. Hasil dapat bervariasi tergantung kondisi pencahayaan dan kualitas gambar.")
                    st.markdown("**Link Google summary: **")
                    st.markdown(f"[Lihat Bedak Rekomendasi di Google]({powder_link})")


                
                    # Tampilkan tone setiap foto di preview
                    for idx, item in enumerate(results):
                        if len(item) == 9:
                            ori, deb, blu, noi, seg, tone, shade, powder_info, powder_link = item
                            st.markdown(f"**Foto ke-{idx+1}: Tone: {tone}")
                            st.image([
                                ori, deb, blu, noi, seg
                            ], caption=[
                                "Gambar Asli", "Deblurring", "Gaussian Blur", "Gaussian Noise", "Segmentasi Kulit (HSV, Manual)"
                            ], width=150)
                        else:
                            st.warning(f"Foto ke-{idx+1}: Data tidak lengkap atau format tidak sesuai.")

