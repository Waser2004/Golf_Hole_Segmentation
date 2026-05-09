import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from golf_hole_segmentation.inference_app import launch_inference_gui

    launch_inference_gui()


if __name__ == "__main__":
    main()
