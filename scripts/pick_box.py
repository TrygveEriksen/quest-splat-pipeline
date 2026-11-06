#!/usr/bin/env python3
"""Interactively pick the subject's bounding box on one frame.

Opens a frame in a window. Drag a rectangle around the object, adjust the
handles if needed, then press Enter (or just close the window). It prints the
box as the --box flag make_masks_sam2.py expects, plus a ready-to-paste SAM 2
command.

    python scripts/pick_box.py                          # work/frames/000000.png
    python scripts/pick_box.py --frame work/frames/000005.png

Needs a GUI (DISPLAY set) and matplotlib with a Tk/Qt backend. If you are on a
headless box, open the PNG in any image viewer instead and read off the
top-left and bottom-right pixel coordinates by hand.
"""
import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="work/frames/000000.png",
                    help="frame to pick on (SAM 2 prompts frame 0 by default)")
    ap.add_argument("--frames-dir", default="work/frames")
    ap.add_argument("--masks-out", default="work/masks")
    ap.add_argument("--checkpoint",
                    default="../sam2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--save", metavar="FILE",
                    help="also write the box as 'x0 y0 x1 y1' to FILE (for "
                         "pipeline.sh); exits non-zero if no box was drawn")
    args = ap.parse_args()

    if not Path(args.frame).is_file():
        ap.error(f"{args.frame} not found -- run stage A (extract_frames) first")

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    img = Image.open(args.frame).convert("RGB")
    W, H = img.size
    arr = np.asarray(img)

    state = {"box": None}

    def onselect(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        box = [max(0, int(round(x0))), max(0, int(round(y0))),
               min(W, int(round(x1))), min(H, int(round(y1)))]
        state["box"] = box
        print(f"[pick] box = {box[0]} {box[1]} {box[2]} {box[3]}  "
              f"({box[2] - box[0]}x{box[3] - box[1]} px)")

    fig, ax = plt.subplots(figsize=(min(12, W / 90), min(12, H / 90)))
    ax.imshow(arr)
    ax.set_title("Drag a box around the object, then press Enter (or close)")

    style = dict(edgecolor="lime", fill=False, linewidth=2)
    try:
        selector = RectangleSelector(ax, onselect, useblit=True, button=[1],
                                     interactive=True, props=style)
    except TypeError:  # older matplotlib names it rectprops
        selector = RectangleSelector(ax, onselect, useblit=True, button=[1],
                                     interactive=True, rectprops=style)

    def on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    box = state["box"]
    if not box or box[2] <= box[0] or box[3] <= box[1]:
        print("[pick] no usable box selected -- nothing to do")
        if args.save:
            sys.exit(1)
        return

    b = " ".join(map(str, box))
    if args.save:
        Path(args.save).write_text(b + "\n")
        print(f"[pick] saved box to {args.save}")
        return
    print("\n=== your box ===")
    print(f"--box {b}")
    print("\n=== make the masks with it ===")
    print(f"conda run -n sam2 --no-capture-output python scripts/make_masks_sam2.py \\")
    print(f"    --frames {args.frames_dir} --out {args.masks_out} \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --box {b}")


if __name__ == "__main__":
    main()
