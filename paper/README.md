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

The paper reports simulation results for reserved routes and Pokhara transfer scenarios. Route and cell-layout inputs, trained policy checkpoints, and complete case-level results are not included in this repository. The PDF therefore documents the results but cannot be rebuilt from the experiment archive using this source tree alone. See [`../docs/DATA_AND_REPRODUCIBILITY.md`](../docs/DATA_AND_REPRODUCIBILITY.md) for details.

This is the identified version. It includes the author information above and is not an anonymized review copy.
