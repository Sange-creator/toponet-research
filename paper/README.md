# Paper

This folder contains the author-identified manuscript, “Graph-Attention Double-DQN for Cellular Handover: A Paired Simulation Evaluation,” by Saange Tamang and Sulav Kandel.

## Files

- `article.pdf` is the paper PDF.
- `article.tex` is the main LaTeX source.
- `paperinfo.tex` contains the author names, affiliation, and contact emails.
- `refs.bib` contains the references.
- `IOEGC.cls` and `pagenum.tex` are needed by the IOEGC template.
- `Graphics/` contains the figures used by the paper.
- `IOEGC19_identified_source.zip` is the source package for editorial use.

## Build the PDF

Install [Tectonic](https://tectonic-typesetting.github.io/) and run:

```sh
cd paper
tectonic article.tex
```

The first build may need an internet connection to download LaTeX packages. The command writes build files next to the source and creates `article.pdf`.

## Results and data

The paper reports simulation results for reserved routes and Pokhara transfer scenarios. A separate, privacy-filtered extract of iPhone 15 dashboard observations from travel along the Prithvi Highway is available in [`../data/highway-dashboard-observation/`](../data/highway-dashboard-observation/); it is field context only, not model-training or evaluation data. The 46-site Kathmandu cell-layout workbook is available in [`../data/kathmandu-osm-cell-layout/`](../data/kathmandu-osm-cell-layout/). Route splits, trained policy checkpoints, and complete simulation case-level results are not included, so the PDF cannot be rebuilt from the full experiment archive using this source tree alone. See [`../docs/DATA_AND_REPRODUCIBILITY.md`](../docs/DATA_AND_REPRODUCIBILITY.md) for details.

This is the identified version. It includes the author information above and is not an anonymized review copy.
