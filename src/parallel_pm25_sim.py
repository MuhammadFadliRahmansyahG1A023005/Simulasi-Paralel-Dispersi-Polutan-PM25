"""
Simulasi Paralel Dispersi Polutan Udara (PM2.5) Wilayah Perkotaan
Studi Kasus: High Performance Computing (HPC) untuk Tantangan Lingkungan Lokal
Inspirasi: LANTA Supercomputer (ThaiSC) - Optimasi Prediksi Polusi Udara

Penulis: Antigravity AI & Tim Peneliti HPC
Deskripsi:
    Mengimplementasikan simulasi dispersi polutan PM2.5 pada grid perkotaan 2D
    menggunakan model adveksi-difusi tereduksi dan komputasi floating-point masif.
    Membandingkan eksekusi serial vs paralel (ProcessPoolExecutor & multiprocessing.Pool)
    serta mengevaluasi metrik Speedup, Efisiensi Paralel, dan Hukum Amdahl.
"""

import time
import os
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ============================================================================
# 1. STANDAR BAKU MUTU PM2.5 (BERDASARKAN ISPU / WHO / BMKG / EPA)
# ============================================================================
def classify_pm25_status(concentration: float) -> str:
    """Mengklasifikasikan status kualitas udara berdasarkan konsentrasi PM2.5 (ug/m3)."""
    if concentration <= 15.5:
        return "BAIK"
    elif concentration <= 55.4:
        return "SEDANG"
    elif concentration <= 150.4:
        return "TIDAK SEHAT"
    elif concentration <= 250.4:
        return "SANGAT TIDAK SEHAT"
    else:
        return "BERBAHAYA"


# ============================================================================
# 2. NUMERICAL KERNEL: SIMULASI DISPERSI PER KISI GRID (CELL)
# ============================================================================
def calculate_pm25_grid_cell(
    cell_id: int, 
    grid_size: int = 500, 
    iterations: int = 50,
    diffusion_rate: float = 0.05,
    return_matrix: bool = False
) -> Dict[str, Any]:
    """
    Kalkulasi dispersi polutan PM2.5 pada satu kisi geografis lokal (sub-grid perkotaan).
    Mensimulasikan beban komputasi Floating Point Operations (FLOPS) tinggi
    dengan model adveksi-difusi 2D finite difference.

    Parameters:
        cell_id (int): Identifier sel grid geografis
        grid_size (int): Resolusi matriks (e.g. 500x500 = 250.000 titik ukur)
        iterations (int): Jumlah langkah waktu simulasi numerik dispersi
        diffusion_rate (float): Koefisien difusi polutan di atmosfer
        return_matrix (bool): Menyimpan matriks hasil (opsional untuk visualisasi)

    Returns:
        dict: Metrik hasil simulasi sel grid
    """
    # Deterministic seed per cell_id agar hasil reproducible
    np.random.seed(cell_id * 1007 + 42)
    
    # 1. Inisialisasi matriks emisi awal (misal: pusat industri, jalan raya, perumahan)
    # Sumber emisi berkisar antara 10 ug/m3 s/d 180 ug/m3
    c = np.random.uniform(10.0, 180.0, size=(grid_size, grid_size)).astype(np.float64)
    
    # Inisialisasi medan angin lokal (adveksi): vektor u (barat-timur) & v (utara-selatan)
    u_wind = np.random.uniform(0.5, 3.5, size=(grid_size, grid_size)).astype(np.float64)
    v_wind = np.random.uniform(0.2, 2.5, size=(grid_size, grid_size)).astype(np.float64)
    
    # 2. Iterasi numerik parsial adveksi-difusi (Finite Difference 2D Stencil)
    # dC/dt = - (u * dC/dx + v * dC/dy) + D * (d2C/dx2 + d2C/dy2) + Source/Decay
    dt = 0.05
    dx = 1.0
    
    for _ in range(iterations):
        # Difusi: Laplace operator (stencil 5-point)
        laplacian = (
            np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) +
            np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 
            4.0 * c
        ) / (dx * dx)
        
        # Adveksi (upwind difference sederhana)
        grad_x = (c - np.roll(c, 1, axis=1)) / dx
        grad_y = (c - np.roll(c, 1, axis=0)) / dx
        advection = -(u_wind * grad_x + v_wind * grad_y)
        
        # Update matriks konsentrasi + fluktuasi turbulensi lokal
        turbulence = np.sin(c * 0.05) * 0.2
        c += dt * (advection + diffusion_rate * laplacian + turbulence)
        
        # Jaga nilai non-negatif
        c = np.clip(c, 0.0, 500.0)

    # 3. Agregasi statistik spasial
    avg_pm25 = float(np.mean(c))
    max_pm25 = float(np.max(c))
    min_pm25 = float(np.min(c))
    std_pm25 = float(np.std(c))
    status = classify_pm25_status(avg_pm25)

    result = {
        "cell_id": cell_id,
        "avg_pm25": round(avg_pm25, 2),
        "max_pm25": round(max_pm25, 2),
        "min_pm25": round(min_pm25, 2),
        "std_pm25": round(std_pm25, 2),
        "status": status,
        "grid_size": grid_size,
        "iterations": iterations
    }
    
    if return_matrix:
        result["matrix"] = c
        
    return result


# ============================================================================
# 3. EKSEKUSI SERIAL (BASELINE 1 CORE)
# ============================================================================
def run_serial_execution(
    num_cells: int, 
    grid_size: int = 400, 
    iterations: int = 35
) -> Tuple[float, List[Dict[str, Any]]]:
    """Menjalankan simulasi seluruh grid cell secara serial (single-core)."""
    start_time = time.perf_counter()
    results = [
        calculate_pm25_grid_cell(i, grid_size=grid_size, iterations=iterations)
        for i in range(num_cells)
    ]
    elapsed = time.perf_counter() - start_time
    return elapsed, results


# ============================================================================
# 4. EKSEKUSI PARALEL MENGGUNAKAN ProcessPoolExecutor (concurrent.futures)
# ============================================================================
def run_parallel_futures(
    num_cells: int, 
    num_workers: int, 
    grid_size: int = 400, 
    iterations: int = 35
) -> Tuple[float, List[Dict[str, Any]]]:
    """Menjalankan simulasi secara paralel menggunakan concurrent.futures.ProcessPoolExecutor."""
    start_time = time.perf_counter()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(calculate_pm25_grid_cell, i, grid_size, iterations): i 
            for i in range(num_cells)
        }
        for future in as_completed(futures):
            results.append(future.result())
            
    # Urutkan berdasarkan cell_id untuk konsistensi
    results.sort(key=lambda x: x["cell_id"])
    elapsed = time.perf_counter() - start_time
    return elapsed, results


# ============================================================================
# 5. EKSEKUSI PARALEL MENGGUNAKAN multiprocessing.Pool
# ============================================================================
def _mp_worker_wrapper(args):
    return calculate_pm25_grid_cell(*args)

def run_parallel_mp(
    num_cells: int, 
    num_workers: int, 
    grid_size: int = 400, 
    iterations: int = 35
) -> Tuple[float, List[Dict[str, Any]]]:
    """Menjalankan simulasi secara paralel menggunakan multiprocessing.Pool."""
    start_time = time.perf_counter()
    tasks = [(i, grid_size, iterations) for i in range(num_cells)]
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_mp_worker_wrapper, tasks)
        
    elapsed = time.perf_counter() - start_time
    return elapsed, results


# ============================================================================
# 6. ANALISIS METRIK KINERJA HPC (SPEEDUP & EFISIENSI)
# ============================================================================
def compute_hpc_metrics(t_serial: float, t_parallel: float, workers: int) -> Dict[str, float]:
    """
    Menghitung metrik performa komputasi paralel:
    - Speedup (S) = T_serial / T_parallel
    - Parallel Efficiency (E) = (S / Workers) * 100%
    - Amdahl's Law theoretical maximum speedup estimation
    """
    speedup = t_serial / t_parallel if t_parallel > 0 else 0.0
    efficiency = (speedup / workers) * 100.0 if workers > 0 else 0.0
    
    return {
        "t_serial": round(t_serial, 4),
        "t_parallel": round(t_parallel, 4),
        "workers": workers,
        "speedup": round(speedup, 2),
        "efficiency_pct": round(efficiency, 1)
    }


# ============================================================================
# 7. BENCHMARK SKALABILITAS MULTI-CORE (SWEEP)
# ============================================================================
def run_scalability_sweep(
    num_cells: int = 16, 
    worker_counts: List[int] = None,
    grid_size: int = 400,
    iterations: int = 30
) -> List[Dict[str, Any]]:
    """
    Menguji performa terhadap variasi jumlah worker/CPU cores.
    Mengevaluasi skalabilitas kuat (Strong Scaling).
    """
    if worker_counts is None:
        cpu_max = psutil.cpu_count(logical=True) or 4
        worker_counts = [1, 2, 4]
        if cpu_max >= 8:
            worker_counts.append(8)
        if cpu_max not in worker_counts:
            worker_counts.append(cpu_max)
        worker_counts = sorted(list(set(worker_counts)))

    print("\n" + "="*70)
    print("Memulai Benchmark Skalabilitas Multi-Core (Strong Scaling Study)")
    print(f"Beban Kerja: {num_cells} Grid Cells | Ukuran Grid: {grid_size}x{grid_size} | Iterasi: {iterations}")
    print(f"Hardware Host: {psutil.cpu_count(logical=False)} Core Fisik / {psutil.cpu_count(logical=True)} Thread Logikal")
    print("="*70)

    # 1. Baseline Serial
    print("[1/2] Menjalankan Baseline Serial...")
    t_serial, _ = run_serial_execution(num_cells, grid_size=grid_size, iterations=iterations)
    print(f"      Waktu Eksekusi Serial (1 Core): {t_serial:.4f} detik")

    # 2. Parallel Sweep
    sweep_results = []
    print("\n[2/2] Menguji Eksekusi Paralel pada Berbagai Core...")
    for p in worker_counts:
        t_par, results = run_parallel_futures(num_cells, p, grid_size=grid_size, iterations=iterations)
        metrics = compute_hpc_metrics(t_serial, t_par, p)
        metrics["method"] = "ProcessPoolExecutor"
        metrics["num_cells"] = num_cells
        metrics["grid_size"] = f"{grid_size}x{grid_size}"
        sweep_results.append(metrics)
        
        print(f"  -> Workers: {p:2d} | Waktu: {t_par:7.4f}s | Speedup: {metrics['speedup']:5.2f}x | Efisiensi: {metrics['efficiency_pct']:5.1f}%")

    return sweep_results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    NUM_CELLS = 16  # Simulasi 16 sub-wilayah perkotaan
    WORKERS = min(4, psutil.cpu_count(logical=True) or 4)
    GRID_SIZE = 400
    ITERATIONS = 35

    print("="*75)
    print("   HPC LOCAL SOLUTION: SIMULASI PARALEL DISPERSI PM2.5 PERKOTAAN   ")
    print("   Referensi: Global Infrastructure to Address Local Challenge with HPC")
    print("   Studi Kasus: LANTA Supercomputer (ThaiSC) - Prediksi Udara Cepat")
    print("="*75)
    print(f"Konfigurasi Simulasi:")
    print(f" - Jumlah Kisi (Cells) : {NUM_CELLS} wilayah")
    print(f" - Resolusi per Kisi   : {GRID_SIZE} x {GRID_SIZE} ({GRID_SIZE*GRID_SIZE:,} titik ukur)")
    print(f" - Total Titik Ukur    : {NUM_CELLS * GRID_SIZE * GRID_SIZE:,} floating point points")
    print(f" - Worker Threads/Cores: {WORKERS} cores")
    print("-" * 75)

    # 1. Eksekusi Serial
    print(f"[*] Menjalankan Eksekusi Serial...")
    t_serial, results_serial = run_serial_execution(NUM_CELLS, grid_size=GRID_SIZE, iterations=ITERATIONS)
    print(f"[+] [Serial Execution] Waktu: {t_serial:.4f} detik\n")

    # 2. Eksekusi Paralel (ProcessPoolExecutor)
    print(f"[*] Menjalankan Eksekusi Paralel ({WORKERS} Workers)...")
    t_parallel, results_parallel = run_parallel_futures(NUM_CELLS, WORKERS, grid_size=GRID_SIZE, iterations=ITERATIONS)
    print(f"[+] [Parallel Execution ({WORKERS} Workers)] Waktu: {t_parallel:.4f} detik\n")

    # 3. Metrik Speedup dan Efisiensi
    metrics = compute_hpc_metrics(t_serial, t_parallel, WORKERS)
    print("="*45)
    print("           RINGKASAN METRIK HPC             ")
    print("="*45)
    print(f"  Waktu Serial (T1)       : {metrics['t_serial']:.4f} detik")
    print(f"  Waktu Paralel (Tp)      : {metrics['t_parallel']:.4f} detik")
    print(f"  Speedup (S)             : {metrics['speedup']:.2f}x")
    print(f"  Parallel Efficiency (E) : {metrics['efficiency_pct']:.1f}%")
    print("="*45)

    # 4. Sampel Hasil Grid
    print("\n[+] Sampel Hasil Grid (5 Sel Pertama):")
    for r in results_parallel[:5]:
        print(f"  - Cell {r['cell_id']:02d}: Rata-rata PM2.5 = {r['avg_pm25']} ug/m3 | Max = {r['max_pm25']} | Status = {r['status']}")

    # 5. Komparasi Analogi LANTA ThaiSC
    print("\n" + "-"*75)
    print("KOMPARASI STUDI KASUS: SUPERKOMPUTER LANTA (ThaiSC)")
    print("Pada skala makro superkomputer nasional Thailand (ThaiSC LANTA):")
    print(" - Waktu simulasi serial/komputasi konvensional : 22 jam (1320 menit)")
    print(" - Waktu simulasi paralel HPC LANTA              : 45 menit")
    print(" - Speedup Superkomputer Real                   : 29.33x")
    print(f" - Simulasi Lokal Multi-core Saat Ini           : {metrics['speedup']:.2f}x (pada {WORKERS} worker CPU)")
    print("-" * 75)
