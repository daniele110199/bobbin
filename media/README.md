# How the demo was made

`demo.gif` is a real run, not a mockup or a re-enactment. The frames are the
bytes the agent actually wrote to the terminal, captured with `script -T` and
replayed with the timings it recorded.

```sh
cp -r evals/fixture-cascade /tmp/bobbin-demo
COLUMNS=100 LINES=30 script -q -T timing.log session.log -c media/session.sh
python3 media/render.py .          # capture -> SVG frames
convert -background none f%05d.svg f%05d.png
ffmpeg -framerate 10 -i f%05d.png -vf "scale=760:-1,split[a][b];\
  [a]palettegen=max_colors=64[p];[b][p]paletteuse" -loop 0 demo.gif
```

Two liberties, both stated in the README caption: playback is sped up 2.6x and
any single pause is capped at 0.9s, because watching a 30B model think is dull.
Nothing is added, removed, or reordered.

The closing checks are the ones the eval suite uses for `cascade-rename`: every
call site updated, and the vendored copy under `node_modules/` untouched. The
model does not pass this case every time; the rates are in the README tables.
