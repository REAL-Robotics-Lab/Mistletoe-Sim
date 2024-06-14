from pathlib import Path
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('isaacsim_path', help='path to isaacsim_directory')
args = parser.parse_args()

isaacsim_path = Path(args.isaacsim_path)

assert isaacsim_path.exists(), f"{args.isaacsim_path} not found"

isaacsim_path = isaacsim_path.resolve()

print(",".join([f"\"{str(f)}\"" for f in isaacsim_path.glob("exts/*")]))