"""
iot_simulation.py — IoT Pipeline & System Integration Simulation
Smart Waste Segregation System

Simulates edge deployment, IoT sensors, bin fill monitoring with alerts.
Fill level: f = 1 - (d_measured / d_max), Alert when f >= 0.75
"""

import os, logging, time, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import config
from utils import set_seed


class BinMonitor:
    """Smart waste bin with ultrasonic fill-level sensor simulation."""

    def __init__(self, bin_id, location, max_depth=config.BIN_MAX_DEPTH_CM):
        self.bin_id = bin_id
        self.location = location
        self.max_depth = max_depth
        self.current_depth = max_depth
        self.fill_history = []
        self.alerts = []
        self.waste_types = []

    @property
    def fill_level(self):
        return 1.0 - (self.current_depth / self.max_depth)

    @property
    def is_alert(self):
        return self.fill_level >= config.ALERT_THRESHOLD

    def add_waste(self, waste_type, volume_cm):
        self.current_depth = max(0, self.current_depth - volume_cm)
        self.waste_types.append(waste_type)
        self.fill_history.append({"fill_level": self.fill_level, "waste_type": waste_type})
        if self.is_alert:
            self.alerts.append(f"Bin {self.bin_id} ALERT: {self.fill_level:.1%}")

    def empty_bin(self):
        self.current_depth = self.max_depth

    def get_status(self):
        return {"bin_id": self.bin_id, "fill_level": round(self.fill_level, 3),
                "is_alert": self.is_alert, "location": self.location}


def simulate_sensor_inputs(n=50):
    """Generate synthetic IoT sensor readings."""
    set_seed()
    categories = {"wet": 0.3, "dry": 0.45, "metal": 0.15, "hazardous": 0.1}
    readings = []
    for i in range(n):
        cat = random.choices(list(categories.keys()), weights=list(categories.values()))[0]
        readings.append({
            "sensor_id": f"S{random.randint(1,20):03d}",
            "moisture": round(random.uniform(10,90) if cat=="wet" else random.uniform(0,30), 1),
            "weight_g": round(random.uniform(50,2000)), "metal_detected": cat=="metal",
            "inferred_category": cat,
        })
    return readings


def simulate_edge_inference(model=None, n_samples=100):
    """Measure edge deployment inference latency."""
    import torch
    if model is None:
        from models import WasteCNN
        model = WasteCNN()
    device = config.DEVICE
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            s = time.perf_counter()
            model(dummy)
            if device.type == 'cuda': torch.cuda.synchronize()
            latencies.append((time.perf_counter()-s)*1000)
    return {"mean_ms": round(np.mean(latencies),2), "fps": round(1000/np.mean(latencies),1)}


def run_bin_simulation(n_bins=config.NUM_BINS, n_steps=config.SIMULATION_STEPS):
    """Run full bin monitoring simulation."""
    set_seed()
    bins = [BinMonitor(i+1, (random.uniform(0,100), random.uniform(0,100))) for i in range(n_bins)]
    for _ in range(n_steps):
        b = random.choice(bins)
        b.add_waste(random.choice(config.CLASSES), random.uniform(2, 15))
        if b.fill_level > 0.9 and random.random() > 0.5: b.empty_bin()
    return bins


def plot_fill_levels(bins, save_path=None):
    """Plot bin fill levels bar chart."""
    plt.figure(figsize=(12, 6))
    ids = [b.bin_id for b in bins]
    levels = [b.fill_level*100 for b in bins]
    colors = ['#e74c3c' if b.is_alert else '#2ecc71' for b in bins]
    plt.bar(ids, levels, color=colors)
    plt.axhline(y=75, color='red', linestyle='--', label='Alert (75%)')
    plt.xlabel('Bin ID'); plt.ylabel('Fill Level (%)'); plt.title('Smart Bin Fill Levels')
    plt.legend(); plt.ylim(0, 105); plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, "bin_fill_levels.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()


def run_full_simulation():
    """Run complete IoT simulation pipeline."""
    logging.info("Running IoT Simulation...")
    readings = simulate_sensor_inputs()
    edge_stats = simulate_edge_inference()
    bins = run_bin_simulation()
    plot_fill_levels(bins)
    logging.info(f"Edge: {edge_stats['mean_ms']}ms, {edge_stats['fps']} FPS")
    return {"sensor_readings": readings, "edge_stats": edge_stats, "bins": bins}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_simulation()
