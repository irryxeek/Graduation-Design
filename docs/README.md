# Docs Structure

`docs/` stores project documents by milestone and purpose.

Primary deliverable: the thesis PDF under `docs/thesis/reports/`.

Current mainline milestone: `thesis/`.
The repository has been aligned so that ATP+WAP processing, training, evaluation,
and thesis writing all point to the same FY-3D 2025 H1 workflow.

- `proposal/`: proposal-stage source materials, references, and opening report files
- `midterm/`: midterm report assets
- `midterm/reports/`: final midterm deliverables
- `midterm/presentation/`: midterm defense slides and outlines
- `midterm/figures/`: figures used in the midterm report or slides
- `midterm/scripts/`: helper scripts for report cleanup, figure insertion, and inspection
- `midterm/workspace/`: local working area for unpacked Office files and temporary extraction outputs
- `defense/`: final defense templates and presentation files
- `thesis/`: thesis draft, outline, progress notes, thesis figures, and ATP+WAP mainline materials
- `thesis/reports/`: exported thesis PDFs or other formal thesis deliverables

Suggested rule: keep formal deliverables in `reports/` or milestone folders, and keep intermediate outputs inside each milestone's `workspace/`.
