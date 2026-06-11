# CS338: SpikeGPT Implementation and Evaluation

This repository contains the implementation, training, and evaluation of the SpikeGPT model, along with a demonstration application and comprehensive project documentation.

## Project Structure

```
CS338/
├── SpikeGPT/           # Core SpikeGPT model implementation
├── data/               # Training and evaluation datasets
│   ├── old_data/       # Legacy dataset versions
│   ├── new_data/       # Current dataset versions
│   └── enwik8_split/   # enwik8 dataset splits
├── Notebooks/          # Jupyter notebooks for experiments
├── Evaluations/        # Model evaluation results and metrics
├── demo/               # Web-based demonstration application
│   ├── backend/        # FastAPI server (demo.py)
│   └── frontend/       # HTML, JavaScript, and CSS files
├── preprocessing/      # Data preprocessing utilities (JSON to binary index)
├── checkpoints/        # Model training checkpoints
└── bao-cao-latex/      # LaTeX project report
```

## Building the Report

The project report is written in LaTeX and requires XeLaTeX for compilation:

```powershell
cd bao-cao-latex
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

**Note:** Run the command twice to ensure proper cross-references and table of contents generation.
