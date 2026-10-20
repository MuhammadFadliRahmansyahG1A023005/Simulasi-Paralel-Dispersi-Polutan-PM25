"""
Modul Visualisasi Hasil Simulasi Paralel Dispersi PM2.5
Menghasilkan grafik performa HPC (Speedup, Efisiensi, Waktu) dan Peta Spasial Dispersi PM2.5.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Pastikan direktori output tersedia
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style visualisasi clean dan modern
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_hpc_scalability(sweep_results: List[Dict[str, Any]], save_path: str = None):
    """
    Menghasilkan grafik Speedup dan Efisiensi Paralel terhadap jumlah CPU Workers.
    Membandingkan performa empiris vs Kecepatan Ideal (Linear Speedup) & Hukum Amdahl.
    """
    df = pd.DataFrame(sweep_results)
    
    workers = df["workers"].values
    speedup = df["speedup"].values
    efficiency = df["efficiency_pct"].values
    t_parallel = df["t_parallel"].values
    t_serial = df["t_serial"].iloc[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # --- 1. Grafik Speedup vs Ideal ---
    ax1.plot(workers, workers, '--', color='#888888', label='Ideal Linear Speedup ($S = p$)', linewidth=2)
    ax1.plot(workers, speedup, 'o-', color='#1f77b4', label='Actual Speedup ($S_{empiris}$)', linewidth=2.5, markersize=8)
    
    # Amdahl's Law Curve approximation (assuming 92% parallel fraction)
    p_frac = 0.92
    amdahl = [1.0 / ((1.0 - p_frac) + (p_frac / w)) for w in workers]
    ax1.plot(workers, amdahl, ':', color='#ff7f0e', label=f"Amdahl's Law ($P={int(p_frac*100)}\\%$)", linewidth=2)
    
    for x, y in zip(workers, speedup):
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#1f77b4')

    ax1.set_title("HPC Speedup Curve\n(Akselerasi Multi-core)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Jumlah Worker (CPU Cores)", fontsize=11)
    ax1.set_ylabel("Speedup ($T_{serial} / T_{paralel}$)", fontsize=11)
    ax1.set_xticks(workers)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Grafik Efisiensi Paralel ---
    ax2.axhline(100, color='#2ca02c', linestyle='--', label='100% Ideal Efficiency', linewidth=1.5)
    ax2.plot(workers, efficiency, 's-', color='#2ca02c', label='Parallel Efficiency ($E$)', linewidth=2.5, markersize=8)
    
    for x, y in zip(workers, efficiency):
        ax2.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold', color='#2ca02c')

    ax2.set_title("Parallel Efficiency\n(Efisiensi Utilisasi Core)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Jumlah Worker (CPU Cores)", fontsize=11)
    ax2.set_ylabel("Efisiensi (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_xticks(workers)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 3. Waktu Eksekusi (Serial vs Paralel) ---
    labels = ['Serial (1 Core)'] + [f'Paralel ({w}C)' for w in workers]
    times = [t_serial] + list(t_parallel)
    colors = ['#d62728'] + ['#17becf' for _ in workers]
    
    bars = ax3.bar(labels, times, color=colors, width=0.55, edgecolor='#333333', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax3.set_title("Waktu Eksekusi Simulasi\n(Execution Time)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_ylabel("Waktu (detik)", fontsize=11)
    ax3.tick_params(axis='x', rotation=25)
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Tolok Ukur Kinerja Paralel Simulasi Dispersi PM2.5\n(Studi Kasus Arsitektur HPC Terdistribusi)", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "hpc_performance_benchmark.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Grafik performa HPC berhasil disimpan ke: {save_path}")
    return save_path


def plot_spatial_pm25_heatmap(grid_results: List[Dict[str, Any]], grid_cols: int = 4, save_path: str = None):
    """
    Menghasilkan visualisasi spasial peta sebaran polutan PM2.5 di wilayah perkotaan.
    Masing-masing subplot merepresentasikan kisi geografis (urban sub-district).
    """
    num_cells = len(grid_results)
    grid_rows = int(np.ceil(num_cells / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3.5, grid_rows * 3.2), dpi=250)
    axes = np.array(axes).reshape(grid_rows, grid_cols)
    
    # Color map ISPU PM2.5 (Hijau -> Kuning -> Oranye -> Merah -> Ungu)
    cmap = plt.colormaps['YlOrRd'] if hasattr(plt, 'colormaps') else plt.cm.get_cmap('YlOrRd')
    norm = mcolors.Normalize(vmin=10.0, vmax=200.0)

    for idx, r in enumerate(grid_results):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]
        
        matrix = r.get("matrix")
        if matrix is None:
            # Fallback visual jika matrix tidak direturn langsung
            grid_size = r.get("grid_size", 200)
            np.random.seed(r["cell_id"] * 1007 + 42)
            matrix = np.random.normal(r["avg_pm25"], r["std_pm25"], size=(grid_size, grid_size))
            
        im = ax.imshow(matrix, cmap=cmap, norm=norm, origin='lower')
        
        # Color tag based on status
        status = r.get("status", "BAIK")
        status_color = {
            "BAIK": "#27ae60",
            "SEDANG": "#f39c12",
            "TIDAK SEHAT": "#e67e22",
            "SANGAT TIDAK SEHAT": "#c0392b",
            "BERBAHAYA": "#8e44ad"
        }.get(status, "#333333")
        
        ax.set_title(f"Cell #{r['cell_id']} ({status})", fontsize=10, fontweight='bold', color=status_color)
        ax.set_xlabel(f"Avg: {r['avg_pm25']:.1f} ug/m3", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Sembunyikan subplot kosong jika ada
    for idx in range(num_cells, grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        axes[row, col].axis('off')

    # Colorbar di sisi kanan
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Konsentrasi PM2.5 ($\mu g/m^3$)', fontsize=11, fontweight='bold')
    
    fig.subplots_adjust(right=0.90, top=0.91, hspace=0.35, wspace=0.25)
    fig.suptitle("Peta Spasial Dispersi Polutan PM2.5 Wilayah Perkotaan (Grid Multiproses)", fontsize=14, fontweight='bold')

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "spatial_pm25_dispersion_map.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Peta spasial PM2.5 berhasil disimpan ke: {save_path}")
    return save_path
