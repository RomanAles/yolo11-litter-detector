import argparse
from litter_detector.datasets import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["land", "underwater", "all"])
    args = parser.parse_args()
    for profile in ("land", "underwater") if args.profile == "all" else (args.profile,):
        prepare(profile)
