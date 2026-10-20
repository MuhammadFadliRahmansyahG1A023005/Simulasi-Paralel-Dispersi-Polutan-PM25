"""
Master Script: Benchmark Komprehensif Simulasi Paralel Dispersi PM2.5
Studi Kasus: Global Infrastructure to Address Local Challenge with HPC (LANTA ThaiSC)

Skrip ini menjalankan:
1. Verifikasi lingkungan hardware (CPU physical cores & logical threads)
2. Eksekusi serial sebagai baseline dasar
3. Eksekusi paralel (ProcessPoolExecutor vs multiprocessing.Pool)
4. Sweep skalabilitas kuat (Strong Scaling: 1, 2, 4, ... Cores)
5. Ekspor metrik performa ke CSV
6. Pembangkitan visualisasi grafik Speedup, Efisiensi & Peta Spasial Polusi 2D
"""

import os
import sys
import psutil
import pandas as pd
from datetime import datetime

# Tambahkan src ke Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, SRC_DIR)

from parallel_pm25_sim import (
    calculate_pm25_grid_cell,
    run_serial_execution,
    run_parallel_futures,
    run_parallel_mp,
    compute_hpc_metrics,
    run_scalability_sweep
)
from visualize_results import plot_hpc_scalability, plot_spatial_pm25_heatmap


def main():
    print("=" * 80)
    print("       SIMULASI PARALEL DISPERSI POLUTAN UDARA (PM2.5) WILAYAH PERKOTAAN       ")
    print("      Studi Kasus: Global Infrastructure to Address Local Challenge with HPC    ")
    print("=" * 80)

    # 1. Hardware Profiling
    p_cores = psutil.cpu_count(logical=False) or 2
    l_cores = psutil.cpu_count(logical=True) or 4
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    
    print(f"\n[INFO HARDWARE]")
    print(f" -> CPU Cores Fisik   : {p_cores} Core")
    print(f" -> CPU Threads Logik : {l_cores} Thread")
    print(f" -> Total RAM Sistem  : {ram_gb} GB")
    print(f" -> Waktu Eksekusi    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    # Parameter Simulasi yang merepresentasikan beban FLOPS masif
    NUM_CELLS = 24       # 24 Zona/Wilayah perkotaan
    GRID_SIZE = 450      # Resolusi per kisi: 450x450 = 202.500 titik ukur per zona
    ITERATIONS = 40      # 40 langkah waktu adveksi-difusi per sel

    # 2. Uji Skalabilitas Multi-Core (Sweep 1, 2, 4, 8 Cores)
    candidate_workers = [1, 2, 4]
    if l_cores >= 8:
        candidate_workers.append(8)
    worker_counts = sorted(list(set(candidate_workers)))

    sweep_results = run_scalability_sweep(
        num_cells=NUM_CELLS,
        worker_counts=worker_counts,
        grid_size=GRID_SIZE,
        iterations=ITERATIONS
    )

    # 3. Komparasi Pool Engine (concurrent.futures vs multiprocessing.Pool)
    target_workers = min(4, l_cores)
    print("\n" + "=" * 70)
    print(f"Komparasi Engine Paralelisasi ({target_workers} Workers):")
    print("=" * 70)
    
    t_serial, _ = run_serial_execution(NUM_CELLS, grid_size=GRID_SIZE, iterations=ITERATIONS)
    t_futures, results_futures = run_parallel_futures(NUM_CELLS, target_workers, grid_size=GRID_SIZE, iterations=ITERATIONS)
    t_mp, results_mp = run_parallel_mp(NUM_CELLS, target_workers, grid_size=GRID_SIZE, iterations=ITERATIONS)

    speedup_futures = t_serial / t_futures
    speedup_mp = t_serial / t_mp
    
    print(f"1. Baseline Serial (1 Core)         : {t_serial:.4f} detik  (Speedup: 1.00x)")
    print(f"2. concurrent.futures (ProcessPool) : {t_futures:.4f} detik  (Speedup: {speedup_futures:.2f}x, Efisiensi: {(speedup_futures/target_workers)*100:.1f}%)")
    print(f"3. multiprocessing.Pool             : {t_mp:.4f} detik  (Speedup: {speedup_mp:.2f}x, Efisiensi: {(speedup_mp/target_workers)*100:.1f}%)")

    # 4. Simpan Hasil ke CSV
    df_metrics = pd.DataFrame(sweep_results)
    csv_path = os.path.join(OUTPUT_DIR, "benchmark_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[+] Hasil benchmark diekspor ke: {csv_path}")

    # 5. Visualisasi Grafik & Heatmap
    print("\n[+] Mengenerate Visualisasi Hasil Simulasi...")
    plot_hpc_scalability(sweep_results)
    
    # Generate peta spasial PM2.5 dengan matrix nyata
    print("[+] Menyiapkan data matriks spasial 2D untuk visualisasi polusi...")
    spatial_data = [
        calculate_pm25_grid_cell(i, grid_size=200, iterations=25, return_matrix=True)
        for i in range(NUM_CELLS)
    ]
    plot_spatial_pm25_heatmap(spatial_data, grid_cols=4)

    # 6. Analogi dan Pembahasan LANTA ThaiSC
    print("\n" + "=" * 80)
    print("                 STUDI KASUS: LANTA SUPERCOMPUTER (ThaiSC)                ")
    print("=" * 80)
    print("Pada studi kasus 'Global Infrastructure to Address Local Challenge with HPC':")
    print(" * Masalah Lokal  : Krisis polusi udara PM2.5 musiman di Thailand utara.")
    print(" * Pendekatan Lama: Komputasi serial/lokal memakan waktu 22 jam untuk 1 run")
    print("                    sehingga peringatan dini polusi selalu terlambat.")
    print(" * Solusi HPC     : Menggunakan superkomputer nasional LANTA (ThaiSC) dengan ribuan core,")
    print("                    memangkas waktu prediksi menjadi hanya 45 menit.")
    print(" * Speedup Nyata  : ~29.33x akselerasi waktu nyata.")
    print("-" * 80)
    print(f" * Simulasi Kita  : Menggunakan {target_workers} Core CPU lokal berhasil mencapai speedup {speedup_futures:.2f}x")
    print(f"                    dengan efisiensi pemanfaatan core sebesar {(speedup_futures/target_workers)*100:.1f}%.")
    print("=" * 80)
    print("\n[SUCCESS] Seluruh rangkaian simulasi, benchmark, dan visualisasi telah selesai!")


if __name__ == "__main__":
    main()
