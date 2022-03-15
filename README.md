# Onion late fusion


## Introduction

This repository performs late fusion on recommendations given by RS based on different content features, as from the Onion Model.

# Folder structure
```
project
│   README.md
│   config.py
│   helpers.py
│   simple_borda.ipynb
│
└───rec_per_user
    │   model=..._audio=..._textual=..._rec_per_user.csv
    │   ...
```
## File description
- `README.md`: this readme.
- `config.py`: configuration file containing paths to rankings shared by one or more notebooks/scripts
- `helpers.py`: module containig functions for handling the `.csv` files to sort out users or to assign and sum up points
- `rec_per_user`: folder containing the individual `.csv` files with the rankings of the models to be fused
- `model=..._audio=..._textual=..._rec_per_user.csv`: files with the rankings of the models to be fused. Those are given as `.csv` with columns `user`, `item`, `rank`.
