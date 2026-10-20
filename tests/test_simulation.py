"""
Unit Tests untuk Simulasi Paralel Dispersi Polutan PM2.5
"""

import unittest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "src")
sys.path.insert(0, SRC_DIR)

from parallel_pm25_sim import (
    classify_pm25_status,
    calculate_pm25_grid_cell,
    compute_hpc_metrics,
    run_serial_execution,
    run_parallel_futures
)

class TestPM25Simulation(unittest.TestCase):
    
    def test_classify_pm25_status(self):
        self.assertEqual(classify_pm25_status(10.0), "BAIK")
        self.assertEqual(classify_pm25_status(15.5), "BAIK")
        self.assertEqual(classify_pm25_status(25.0), "SEDANG")
        self.assertEqual(classify_pm25_status(55.4), "SEDANG")
        self.assertEqual(classify_pm25_status(75.0), "TIDAK SEHAT")
        self.assertEqual(classify_pm25_status(200.0), "SANGAT TIDAK SEHAT")
        self.assertEqual(classify_pm25_status(300.0), "BERBAHAYA")

    def test_calculate_pm25_grid_cell_output_structure(self):
        res = calculate_pm25_grid_cell(cell_id=1, grid_size=50, iterations=5, return_matrix=True)
        self.assertEqual(res["cell_id"], 1)
        self.assertIn("avg_pm25", res)
        self.assertIn("status", res)
        self.assertIn("matrix", res)
        self.assertEqual(res["matrix"].shape, (50, 50))
        self.assertTrue(res["avg_pm25"] >= 0)

    def test_compute_hpc_metrics(self):
        metrics = compute_hpc_metrics(t_serial=10.0, t_parallel=2.5, workers=4)
        self.assertEqual(metrics["speedup"], 4.0)
        self.assertEqual(metrics["efficiency_pct"], 100.0)

        # Realistic scaling case
        metrics2 = compute_hpc_metrics(t_serial=10.0, t_parallel=3.33333333333, workers=4)
        self.assertAlmostEqual(metrics2["speedup"], 3.0, places=1)
        self.assertAlmostEqual(metrics2["efficiency_pct"], 75.0, places=1)

    def test_serial_vs_parallel_execution(self):
        # Small scale quick integration test
        num_cells = 4
        t_serial, res_serial = run_serial_execution(num_cells, grid_size=40, iterations=5)
        t_par, res_par = run_parallel_futures(num_cells, num_workers=2, grid_size=40, iterations=5)
        
        self.assertEqual(len(res_serial), num_cells)
        self.assertEqual(len(res_par), num_cells)
        # Results should be identical because deterministic random seeds are used per cell_id
        for s, p in zip(res_serial, res_par):
            self.assertEqual(s["cell_id"], p["cell_id"])
            self.assertEqual(s["avg_pm25"], p["avg_pm25"])
            self.assertEqual(s["status"], p["status"])


if __name__ == "__main__":
    unittest.main()
