# Simulasi Paralel Dispersi Polutan Udara (PM2.5) Wilayah Perkotaan
## Studi Kasus: Global Infrastructure to Address Local Challenge with HPC (LANTA Supercomputer - ThaiSC)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![HPC Concurrency](https://img.shields.io/badge/Concurrency-multiprocessing%20%7C%20concurrent.futures-green.svg)]()
[![Domain](https://img.shields.io/badge/Domain-Atmospheric%20Dispersion%20%26%20HPC-orange.svg)]()

---

## 1. Latar Belakang & Studi Kasus HPC

### 1.1 Masalah Lokal: Prediksi Kualitas Udara Perkotaan (PM2.5)
Partikulat halus berukuran kurang dari 2.5 mikrometer (**PM2.5**) merupakan salah satu polutan udara paling berbahaya bagi kesehatan publik di wilayah perkotaan (gangguan respirasi, kardiovaskular, dan karsinogenik). Memprediksi pemetaan dan dinamika pergerakan polutan udara di tingkat lokal (kabupaten/kota/kecamatan) membutuhkan perhitungan numerik matriks kisi (*grid matrix*) geografis yang masif. 

Model atmosferik harus memperhitungkan:
1. **Adveksi**: Pergerakan polutan oleh vektor medan angin horizontal ($u, v$).
2. **Difusi**: Penyebaran konsentrasi zat dari konsentrasi tinggi ke rendah akibat turbulensi atmosferik ($\nabla^2 C$).
3. **Emisi Lokal & Kimia**: Emisi kendaraan bermotor, kawasan industri, dan fluktuasi fotokimia.

Perhitungan numerik diferensial parsial (*partial differential equations / PDE*) pada jutaan titik kisi membutuhkan operasi bilangan riil (*Floating Point Operations / FLOPS*) yang sangat masif.

---

### 1.2 Studi Kasus: Superkomputer LANTA (ThaiSC)
Proyek ini mengadopsi prinsip yang dipaparkan dalam materi **"Global Infrastructure to Address Local Challenge with HPC"**:
* **Tantangan Nyata**: Wilayah Thailand utara (seperti Chiang Mai) menghadapi krisis kabut asap polusi musiman (*seasonal haze / biomass burning*). Model perkiraan kualitas udara konvensional membutuhkan waktu **22 jam** untuk menghasilkan satu putaran prediksi harian pada sistem komputasi serial atau kluster lokal skala kecil. Akibatnya, peringatan dini (*early warning*) datang terlambat kepada publik.
* **Solusi HPC**: Menggunakan infrastruktur superkomputer nasional **LANTA** di *Thailand National Science and Technology Development Agency (NSTDA) / ThaiSC*. Superkomputer ini memanfaatkan ribuan core CPU dan akselerator GPU berkecepatan tinggi terinterkoneksi jaringan latensi rendah.
* **Dampak**: Waktu simulasi pemodelan dipangkas drastis dari **22 jam menjadi hanya 45 menit** (akselerasi / *speedup* riil mencapai **~29.33x**). Hal ini memungkinkan mitigasi bencana polusi dan peringatan dini dikirim tepat waktu sebelum kualitas udara memburuk.

---

## 2. Landasan Teori Komputasi Paralel

### 2.1 Persamaan Dispersi Adveksi-Difusi 2D
Simulasi dispersi polutan pada setiap sel grid wilayah dimodelkan dengan persamaan transport atmosferik:
$$\frac{\partial C}{\partial t} = - \left(u \frac{\partial C}{\partial x} + v \frac{\partial C}{\partial y}\right) + D \left(\frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2}\right) + S(x, y, t)$$
Di mana:
* $C(x, y, t)$: Konsentrasi polutan PM2.5 ($\mu g/m^3$)
* $u, v$: Komponen kecepatan vektor angin zonal dan meridional
* $D$: Koefisien difusi atmosferik
* $S$: Sumber emisi dan deposisi

Persamaan di atas diselesaikan secara numerik menggunakan metode **Finite Difference (Euler forward time, 5-point Laplacian stencil)** pada kisi diskrit resolusi tinggi.

---

### 2.2 Metrik Kinerja HPC

1. **Waktu Eksekusi Serial ($T_1$ / $T_{\text{serial}}$)**:
   Waktu total yang dibutuhkan untuk menyelesaikan seluruh sel komputasi menggunakan 1 unit core CPU secara berurutan.

2. **Waktu Eksekusi Paralel ($T_p$ / $T_{\text{parallel}}$)**:
   Waktu total yang dibutuhkan saat beban komputasi didistribusikan ke $p$ buah pekerja (*workers / cores*).

3. **Speedup ($S$)**:
   Rasio percepatan komputasi paralel dibandingkan baseline serial:
   $$S(p) = \frac{T_1}{T_p}$$
   *Ideal*: $S(p) = p$ (*Linear Speedup*).

4. **Efisiensi Paralel ($E$)**:
   Persentase efektivitas pemanfaatan core CPU:
   $$E(p) = \frac{S(p)}{p} \times 100\% = \frac{T_1}{p \cdot T_p} \times 100\%$$
   *Nilai ideal*: 100%. Pada sistem riil, efisiensi menurun seiring bertambahnya core akibat *overhead* komunikasi antar-proses, sinkronisasi, dan pembagian memori.

5. **Hukum Amdahl (Amdahl's Law)**:
   Menyatakan batas teoritis maksimum speedup yang dapat dicapai:
   $$S_{\text{max}}(p) = \frac{1}{(1 - P) + \frac{P}{p}}$$
   Di mana $P$ adalah fraksi program yang dapat diparalelkan ($0 \le P \le 1$).

---

## 3. Struktur Direktori Proyek

```
Simulasi Paralel Dispersi Polutan Udara (PM2.5) Wilayah Perkotaan/
├── README.md                     # Dokumentasi komprehensif & panduan HPC
├── requirements.txt              # Daftar dependensi Python
├── run_benchmark.py              # Master runner: benchmark, ekspor CSV & grafik
├── src/
│   ├── parallel_pm25_sim.py      # Kernel numerik, ProcessPoolExecutor & mp.Pool
│   └── visualize_results.py      # Visualisasi grafik Speedup, Efisiensi & Peta Spasial
├── tests/
│   └── test_simulation.py        # Unit testing numerik, status ISPU & konkurensi
└── output/                       # Folder artefak hasil eksekusi (auto-generated)
    ├── benchmark_metrics.csv     # Data mentah benchmark komparasi core
    ├── hpc_performance_benchmark.png # Grafik Speedup, Efisiensi & Eksekusi
    └── spatial_pm25_dispersion_map.png # Peta heatmap sebaran PM2.5 2D
```

---

## 4. Instalasi & Prasyarat

Pastikan Python 3.9+ telah terpasang di sistem. Pasang dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

---

## 5. Cara Menjalankan Program

### 5.1 Menjalankan Simulasi Dasar
Untuk menjalankan simulasi dasar serial vs paralel:
```bash
python src/parallel_pm25_sim.py
```

### 5.2 Menjalankan Uji Benchmark Lengkap & Visualisasi
Untuk menjalankan seluruh pengujian skalabilitas (Strong Scaling), membandingkan `concurrent.futures` vs `multiprocessing.Pool`, mengekspor data ke CSV, dan menghasilkan visualisasi:
```bash
python run_benchmark.py
```

### 5.3 Menjalankan Unit Testing
Untuk memverifikasi kebenaran matematika, klasifikasi baku mutu PM2.5, dan determinisme hasil:
```bash
python tests/test_simulation.py
```

---

## 6. Standar Klasifikasi Indeks Kualitas Udara (PM2.5)

Berdasarkan baku mutu konsentrasi PM2.5 24-jam (ISPU KLHK / BMKG / WHO):
| Rentang Konsentrasi ($\mu g/m^3$) | Kategori Status | Dampak Kesehatan |
|---|---|---|
| **0 - 15.5** | **BAIK** | Tingkat mutu udara sangat baik, tidak berefek negatif. |
| **15.6 - 55.4** | **SEDANG** | Kualitas udara dapat diterima, sensitif bagi kelompok rentan. |
| **55.5 - 150.4** | **TIDAK SEHAT** | Merugikan manusia, hewan, dan merusak tanaman. |
| **150.5 - 250.4** | **SANGAT TIDAK SEHAT** | Meningkatkan sensitivitas pada populasi berisiko tinggi. |
| **> 250.4** | **BERBAHAYA** | Tingkat berbahaya secara umum bagi seluruh populasi. |

---

## 7. Hasil & Pembahasan

Dari pengujian *strong scaling* pada lingkungan komputasi multi-core:
1. **Peningkatan Kinerja (*Speedup*)**: Pemrosesan paralel multi-worker secara signifikan mempercepat perhitungan adveksi-difusi matriks besar dibandingkan eksekusi serial 1-core.
2. **Efisiensi Paralel**: Pemanfaatan core optimal diperoleh pada alokasi core fisik CPU. Peningkatan jumlah worker melebihi core fisik menuju hyper-threads logikal akan mulai mengalami hukum hasil yang semakin berkurang (*diminishing returns*) sebagaimana diprediksi oleh Hukum Amdahl.
3. **Relevansi Studi Kasus LANTA**: Simulasi ini membuktikan secara empiris bahwa teknik dekomposisi domain spasial (*spatial domain decomposition*) adalah kunci keberhasilan superkomputer modern dalam mereduksi waktu simulasi bencana dan polusi udara dari orde puluhan jam menjadi hitungan menit.
