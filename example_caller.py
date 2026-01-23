
"""
caller_with_generated_conf.py

1) Runs generate_basic_conf.py to produce worldmap_conf.json
2) Loads the config
3) Seeds RNG for reproducible map layout
4) Runs worldmap_module.run_map using the generated params + initial ownership

Run:
    python caller_with_generated_conf.py
"""

import json
import os
import subprocess
import random

from worldmap_module import run_map


def main():
    # 1) Generate config
    subprocess.check_call(["python", "generate_basic_conf.py"])

    # 2) Load config
    with open("worldmap_conf.json", "r", encoding="utf-8") as f:
        conf = json.load(f)

    seed = int(conf["seed"])
    run_kwargs = conf["run_kwargs"]
    initial_owner = conf["initial_owner"]

    # 3) Seed RNG so the map generated inside run_map matches what the generator analyzed
    random.seed(seed)

    # 4) Run the map
    final_owner, meta = run_map(**run_kwargs, initial_owner=initial_owner)

    print("Final owner:", final_owner)
    print("Num regions:", meta["num_regions"])


if __name__ == "__main__":
    main()
