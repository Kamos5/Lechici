"""
generate_basic_conf.py

Generates a basic configuration for worldmap_module with:
- num_regions = 25
- 4 players (+ None)
- each player starts with at least 3 connected regions
- those starter clusters are placed near the four corners of the map

Outputs a JSON file (worldmap_conf.json) with:
{
  "seed": <int>,
  "run_kwargs": {...kwargs for run_map...},
  "initial_owner": [ ... length num_regions ... ],
  "clusters": {"0":[...], "1":[...], "2":[...], "3":[...]}
}

Run:
    python generate_basic_conf.py
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Set

import numpy as np

import worldmap_module as wm


def build_adjacency(assign: np.ndarray, num_regions: int) -> List[Set[int]]:
    h, w = assign.shape
    adj: List[Set[int]] = [set() for _ in range(num_regions)]
    for y in range(h):
        row = assign[y]
        row_d = assign[y + 1] if y < h - 1 else None
        for x in range(w):
            a = int(row[x])
            if x < w - 1:
                b = int(row[x + 1])
                if a != b:
                    adj[a].add(b)
                    adj[b].add(a)
            if y < h - 1:
                c = int(row_d[x])
                if a != c:
                    adj[a].add(c)
                    adj[c].add(a)
    return adj


def region_centroids(assign: np.ndarray, num_regions: int) -> np.ndarray:
    h, w = assign.shape
    ys, xs = np.indices((h, w))
    flat = assign.ravel()
    areas = np.bincount(flat, minlength=num_regions).astype(np.float32)
    sum_x = np.bincount(flat, weights=xs.ravel(), minlength=num_regions).astype(np.float32)
    sum_y = np.bincount(flat, weights=ys.ravel(), minlength=num_regions).astype(np.float32)
    safe = np.maximum(areas, 1.0)
    cx = sum_x / safe
    cy = sum_y / safe
    return np.stack([cx, cy], axis=1)  # (num_regions, 2) in (x,y)


def pick_connected_cluster(start: int, adj: List[Set[int]], k: int, forbidden: Set[int]) -> List[int]:
    cluster: List[int] = []
    queue = [int(start)]
    seen = {int(start)}
    while queue and len(cluster) < k:
        u = int(queue.pop(0))
        if u in forbidden:
            continue
        cluster.append(u)

        nbrs = [int(v) for v in adj[u] if int(v) not in seen]
        nbrs.sort(key=lambda v: len(adj[v]))  # compact-ish
        for v in nbrs:
            seen.add(v)
            queue.append(v)

    return cluster[:k] if len(cluster) >= k else []


def find_corner_clusters(assign: np.ndarray, num_regions: int, k: int = 3):
    adj = build_adjacency(assign, num_regions)
    cent = region_centroids(assign, num_regions)

    h, w = assign.shape
    corners = {
        0: (0.0, 0.0),          # top-left
        1: (w - 1.0, 0.0),      # top-right
        2: (0.0, h - 1.0),      # bottom-left
        3: (w - 1.0, h - 1.0),  # bottom-right
    }

    ranked: Dict[int, List[int]] = {}
    for pid, (cx, cy) in corners.items():
        d = np.sqrt((cent[:, 0] - cx) ** 2 + (cent[:, 1] - cy) ** 2)
        ranked[pid] = [int(x) for x in np.argsort(d).astype(int)]

    taken: Set[int] = set()
    clusters: Dict[int, List[int]] = {}

    for pid in range(4):
        ok = False
        for r in ranked[pid][:10]:
            if r in taken:
                continue
            cl = pick_connected_cluster(r, adj, k=k, forbidden=taken)
            if cl:
                clusters[int(pid)] = [int(x) for x in cl]
                taken.update(cl)
                ok = True
                break
        if not ok:
            return None

    return clusters


def main():
    run_kwargs = dict(
        player_count=4,
        num_regions=25,
        map_w=400,
        map_h=300,
        width=800,
        height=600,
        background_path="background.png",
        tile_alpha=128,   # 50% opacity
        none_alpha=204,   # 80% opacity
    )

    max_attempts = 5000

    for _ in range(max_attempts):
        seed = random.randint(1, 2_000_000_000)

        # IMPORTANT: worldmap_module uses Python's random for region site placement
        random.seed(seed)

        cfg = wm.MapConfig(**run_kwargs)
        assign = wm._generate_regions(cfg, cfg.map_w, cfg.map_h, run_kwargs["num_regions"])

        clusters = find_corner_clusters(assign, run_kwargs["num_regions"], k=3)
        if clusters is None:
            continue

        none_idx = 4  # for player_count=4 => 4 players + None at index 4

        initial_owner = [int(none_idx)] * int(run_kwargs["num_regions"])
        for pid, regs in clusters.items():
            for r in regs:
                initial_owner[int(r)] = int(pid)

        clusters_json = {str(int(pid)): [int(x) for x in regs] for pid, regs in clusters.items()}

        out = {
            "seed": int(seed),
            "run_kwargs": run_kwargs,
            "initial_owner": [int(x) for x in initial_owner],
            "clusters": clusters_json,
        }

        with open("worldmap_conf.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print("Wrote worldmap_conf.json")
        print("seed:", seed)
        print("clusters:", clusters_json)
        return

    raise SystemExit(f"Failed to find a suitable seed in {max_attempts} attempts.")


if __name__ == "__main__":
    main()
