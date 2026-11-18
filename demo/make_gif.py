"""Assemble demo/demo.gif from the frames saved by demo/record_gif.mjs."""
import json
from pathlib import Path

from PIL import Image, ImageChops

DEMO = Path(__file__).resolve().parent
FRAMES = DEMO / "e2e_out" / "frames"
WIDTH = 900


def main():
    meta = json.loads((FRAMES / "frames.json").read_text())
    imgs, durs = [], []
    for f in meta:
        im = Image.open(DEMO / f["path"]).convert("RGB")
        im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
        if imgs and not ImageChops.difference(im, imgs[-1]).getbbox():
            durs[-1] += f["ms"]                     # identical to the previous frame: just show it longer
            continue
        imgs.append(im)
        durs.append(f["ms"])
    # One shared palette (built from a few frames) keeps colors stable across frames and the file small.
    sample = Image.new("RGB", (WIDTH, imgs[0].height * 4))
    for i, k in enumerate([0, len(imgs) // 3, 2 * len(imgs) // 3, len(imgs) - 1]):
        sample.paste(imgs[k], (0, i * imgs[0].height))
    pal = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    out = DEMO / "demo.gif"
    q[0].save(out, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"{out}: {len(q)} frames, {sum(durs) / 1000:.1f} s, {out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
