# Portfolio — Hoang Huy Dang

Personal portfolio site. Two entrances, same content:

- **[index.html](index.html)** — static one-page portfolio. Facts in ten seconds.
- **[desktop.html](desktop.html)** — interactive KDE Plasma-style desktop environment,
  with draggable/resizable windows, a taskbar, an application launcher, and a
  working terminal (`help`, `neofetch`, `open <app>`).

No framework, no build step, no dependencies — hand-written HTML, CSS, and vanilla
JS. The window manager (pointer-capture dragging, resize, focus stacking, z-order,
minimise/maximise) is about 200 lines of plain JS.

The only external request is a Google Fonts stylesheet; without it the pages fall
back to system fonts.

## Run locally

Open either file directly in a browser, or serve the folder:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

## Deploy

Static hosting, nothing to build. On GitHub Pages, push this folder to a repo and
enable Pages on the `main` branch, root directory. `.nojekyll` disables Jekyll
processing, which this site does not need.
