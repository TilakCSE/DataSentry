# DataSentry AI paper workspace

## Current status

`main.tex` is a compiled IEEE-format **working draft**, dated 30 October 2026.
It cites 33 distinct research papers plus the source dataset entry. The
student-data results are a five-split preliminary pilot. The multi-dataset
study, non-random missingness, numeric corruption, and neural comparison remain
to be run before this is a submission-ready paper. Do not remove the working
draft notice or describe the pilot as the final evidence until those runs exist.

`pilot_rmse_delta.pdf` is a vector figure generated from
`../experiments/pilot_student_results.csv` by
`../experiments/make_pilot_figure.py`. The paper's workflow diagram, related-work
table, dataset table, and pilot table are labelled in `main.tex`.

## Mendeley workflow

The reference library is `references.bib`. Mendeley Reference Manager supports
BibTeX import: **+ Add new > Import library > select `references.bib`**. Create
a DataSentry collection and check each imported entry's authors, title, year,
venue, DOI, and duplicates against the linked primary source. After corrections,
use **File > Export All > BibTeX** in Mendeley Reference Manager Desktop and
replace `references.bib` with the exported file, preserving citation keys used
by `main.tex` (or update those keys in the TeX source). Recompile and inspect
the IEEE-numbered bibliography. Mendeley Cite in Word is a separate route; the
current manuscript uses LaTeX/BibTeX and the library is prepared for Mendeley.

Mendeley's official import and export instructions:

* https://static.mendeley.com/md-stitch/releases/live/02-adding-references.e005845c.html
* https://static.mendeley.com/md-stitch/releases/live/08-exporting-references.cb27d2e0.html

No signed-in Mendeley session was available to this task, so the library has
been prepared for import rather than claimed as already synchronized to an
account.

## Paper requirements and open checks

* Keep the detailed section order from the supplied screenshot; meet the
  user's stricter minimum of 10 pages and 30 distinct cited papers.
* The typed and photographed section-length instructions conflict. Favor
  substance and the typed minimums; do not add filler to force exact lengths.
* Confirm the official student CSV against the pilot's documented mirror.
* Complete the multi-dataset benchmark and add actual results and limitations.
* Check final page count, figure placement, references, affiliation, and IEEE
  style in the rendered PDF before submission.
* The final paper and project must contain no reference to any other project.
