# DataSentry AI paper and evidence

`DataSentry_AI_Research_Paper.pdf` is the visually checked 10-page IEEE two-column manuscript compiled from `main.tex` on 30 October 2026. It reports the completed three-dataset study, including outcomes that favor, tie, and disfavor the training-guided policy. The source cites 33 distinct research papers and one dataset entry. `references.bib` contains all 34 cited records. The paper makes no claim of publication or guaranteed marks.

The `results/` folder contains run-level CSVs and adjacent JSON manifests with input hashes and the exact study grid. `study_summary.csv`, `mcar_curves.pdf`, and `policy_heatmap.pdf` are generated from those results by `../experiments/summarize_study.py`. `pilot_rmse_delta.pdf` is an older, separately labeled pilot artifact and is not part of the final paper.

## Rebuild

From the repository root, install the Python package and run the three commands in the root README to recreate the full study. Then run:

```powershell
.\.venv\Scripts\python.exe experiments\summarize_study.py --input-dir paper\results --output-dir paper
```

Compile `main.tex` with an IEEEtran-capable LaTeX environment and BibTeX. The committed PDF is the reviewed reference render. The summary script checks the complete 2,160-arm grid before it writes figures.

## Mendeley Reference Manager

The paper uses IEEE-numbered BibTeX citations. To meet the course requirement for Mendeley Reference Manager, sign in to the desktop app, choose **+ Add new > Import library**, and select `references.bib`. Put the imported records in a DataSentry collection. Verify author names, title, year, venue, DOI/URL, and duplicates against the linked primary publications. Then use **File > Export All > BibTeX** and update `references.bib`; preserve citation keys used by `main.tex`, or update the TeX source and recompile. Mendeley Cite for Word is a separate workflow and is not needed for this LaTeX paper.

Official instructions: [import](https://static.mendeley.com/md-stitch/releases/live/02-adding-references.e005845c.html) and [export](https://static.mendeley.com/md-stitch/releases/live/08-exporting-references.cb27d2e0.html). No signed-in Mendeley session was available during preparation, so the current library is **ready for import**, not claimed to be synchronized with an account.
