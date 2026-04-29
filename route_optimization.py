"""
route_optimization.py — Collection Route Optimization
Smart Waste Segregation System

Uses XGBoost for priority prediction and nearest-neighbor for route planning.
"""

import os, logging, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils import set_seed


def generate_bin_data(bins=None, n_bins=config.NUM_LOCATIONS):
    """Generate bin location and fill data for route optimization."""
    set_seed()
    if bins is not None:
        data = []
        for b in bins:
            data.append({
                "bin_id": b.bin_id, "x": b.location[0], "y": b.location[1],
                "fill_level": b.fill_level, "hours_since_collection": random.uniform(1, 72),
                "avg_daily_fill_rate": random.uniform(0.05, 0.3),
            })
        return data
    
    data = []
    for i in range(n_bins):
        data.append({
            "bin_id": i+1,
            "x": random.uniform(0, config.GRID_SIZE[0]),
            "y": random.uniform(0, config.GRID_SIZE[1]),
            "fill_level": random.uniform(0.1, 0.95),
            "hours_since_collection": random.uniform(1, 72),
            "avg_daily_fill_rate": random.uniform(0.05, 0.3),
        })
    return data


def train_priority_model(bin_data):
    """Train XGBoost to predict collection priority score."""
    from xgboost import XGBRegressor
    
    X = np.array([[b["fill_level"], b["hours_since_collection"],
                    b["avg_daily_fill_rate"]] for b in bin_data])
    # Priority = weighted combination (higher = more urgent)
    y = np.array([b["fill_level"]*0.5 + (b["hours_since_collection"]/72)*0.3 +
                   b["avg_daily_fill_rate"]*0.2 for b in bin_data])
    
    model = XGBRegressor(n_estimators=100, max_depth=5, random_state=config.RANDOM_SEED)
    model.fit(X, y)
    predictions = model.predict(X)
    
    for b, p in zip(bin_data, predictions):
        b["priority"] = float(p)
    
    logging.info("XGBoost priority model trained.")
    return model, bin_data


def nearest_neighbor_route(bin_data, start=(0, 0)):
    """Nearest-neighbor heuristic for route optimization."""
    # Sort by priority (high first), then take top bins needing collection
    urgent = [b for b in bin_data if b.get("priority", b["fill_level"]) > 0.4]
    if not urgent:
        urgent = bin_data
    
    route = []
    current = start
    remaining = list(urgent)
    
    while remaining:
        distances = [np.sqrt((b["x"]-current[0])**2 + (b["y"]-current[1])**2) for b in remaining]
        nearest_idx = np.argmin(distances)
        nearest = remaining.pop(nearest_idx)
        route.append(nearest)
        current = (nearest["x"], nearest["y"])
    
    total_dist = 0
    prev = start
    for b in route:
        total_dist += np.sqrt((b["x"]-prev[0])**2 + (b["y"]-prev[1])**2)
        prev = (b["x"], b["y"])
    
    return route, total_dist


def plot_route(route, bin_data, save_path=None):
    """Visualize optimized collection route on 2D map."""
    plt.figure(figsize=(10, 10))
    
    # Plot all bins
    for b in bin_data:
        color = '#e74c3c' if b["fill_level"] >= 0.75 else '#f39c12' if b["fill_level"] >= 0.5 else '#2ecc71'
        plt.scatter(b["x"], b["y"], c=color, s=b["fill_level"]*300+50,
                   edgecolors='black', linewidth=0.5, zorder=5)
        plt.annotate(f'B{b["bin_id"]}\n{b["fill_level"]:.0%}',
                    (b["x"], b["y"]), textcoords="offset points",
                    xytext=(0,12), ha='center', fontsize=8)
    
    # Plot route
    if route:
        rx = [0] + [b["x"] for b in route]
        ry = [0] + [b["y"] for b in route]
        plt.plot(rx, ry, 'b--', linewidth=1.5, alpha=0.7, zorder=3)
        for i in range(len(rx)-1):
            plt.annotate('', xy=(rx[i+1], ry[i+1]), xytext=(rx[i], ry[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    plt.scatter([0], [0], c='blue', marker='s', s=200, zorder=10, label='Depot')
    plt.xlabel('X Coordinate'); plt.ylabel('Y Coordinate')
    plt.title('Optimized Waste Collection Route', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, "collection_route.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
    logging.info(f"Route plot saved: {save_path}")


def run_route_optimization(bins=None):
    """Run complete route optimization pipeline."""
    logging.info("Running Route Optimization...")
    bin_data = generate_bin_data(bins)
    model, bin_data = train_priority_model(bin_data)
    route, distance = nearest_neighbor_route(bin_data)
    
    # Compare with unoptimized (sequential) route
    seq_dist = 0
    prev = (0, 0)
    for b in bin_data:
        seq_dist += np.sqrt((b["x"]-prev[0])**2 + (b["y"]-prev[1])**2)
        prev = (b["x"], b["y"])
    
    improvement = ((seq_dist - distance) / seq_dist) * 100
    logging.info(f"Optimized distance: {distance:.1f} vs Sequential: {seq_dist:.1f} "
                 f"({improvement:.1f}% improvement)")
    
    plot_route(route, bin_data)
    return {"route": route, "distance": distance, "sequential_distance": seq_dist,
            "improvement_pct": improvement, "model": model}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_route_optimization()
