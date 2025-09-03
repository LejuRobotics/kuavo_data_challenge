import json
import os
import shutil
import subprocess
import rosbag
from pathlib import Path
from typing import Any

def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def reindex_rosbag(bag_file)->str:
    bag_file = str(bag_file)
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            return bag_file
    except rosbag.bag.ROSBagException as e:
        print(f"Error reading '{bag_file}': {e}")
    
    # bag is corrupted.
    try:
        print(f"Warning: The bag file '{bag_file}' is corrupted, reindexing...")
        command = [
            "rosbag",
            "reindex",
            bag_file
            ]
        subprocess.run(command, check=True)
        if bag_file.endswith(".bag.active"):
            base_name = bag_file.replace(".bag.active", "")
            rosbag_orig_file = f"{base_name}.bag.orig.active"
        elif bag_file.endswith(".bag"):
            base_name = bag_file.replace(".bag", "")
            rosbag_orig_file = f"{bag_file}.orig.bag"
        if os.path.exists(rosbag_orig_file):
            os.remove(rosbag_orig_file)
        if os.path.exists(bag_file):
            rosbag_file = f"{base_name}.bag"
            shutil.move(bag_file, rosbag_file)
            return rosbag_file
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error reindexing bag file: {e}")
        return None