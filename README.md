# Portfolio — Hoang Huy Dang

Personal portfolio site. Two entrances, same content:

- **[index.html](index.html)** — static one-page portfolio. Facts in ten seconds.
- **[desktop.html](desktop.html)** — interactive KDE Plasma-style desktop environment,
  with draggable/resizable windows, a taskbar, an application launcher, and a
  working terminal (`help`, `neofetch`, `open <app>`).

No framework, no build step, no dependencies — hand-written HTML, CSS, and vanilla
JS. The window manager (pointer-capture dragging, resize, focus stacking, z-order,
minimise/maximise) is about 200 lines of plain JS.

## Documents

Desktop mode has a **Documents** folder that opens as a two-level tree, split into
two sections — university projects and side projects — with one folder per project.
Clicking a document opens it in an in-window PDF viewer (`<iframe>` on the browser's
built-in plugin) with an "Open in new tab" escape hatch and a breadcrumb back to the
root.

| Section | Folder | Document | What it is |
|---------|--------|----------|------------|
| University | Master Project | `learning-curve-extrapolation-study-project.pdf` | Study project slides, 41 pages |
| University | AutoML Lab Project | `switching-optimizers-poster.pdf` | A0 research poster, 1 page |
| University | Seminar 1 | `dino-emerging-properties-seminar.pdf` | DINO seminar talk, 47 slides |
| University | Seminar 2 | `lhopt-learned-optimizers-seminar.pdf` | LHOPT seminar talk, 26 slides |
| Side | LLM code-review evaluation | — | in progress, nothing published yet |

The two seminar decks were authored in PowerPoint and converted with LibreOffice
(`soffice --headless --convert-to pdf`); the `.pptx` originals are not in the repo.

To add a project, append one entry to `UNI_FOLDERS` or `SIDE_FOLDERS` in
[desktop.html](desktop.html) — a folder id, name, description, and its files (window
id, path, display name, metadata line). The folder window, the PDF viewer windows,
and the `docs` terminal listing all derive from it. A folder with an empty `files`
array renders as a dashed, non-clickable placeholder; give it a file and it becomes
browsable with no other change.

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
