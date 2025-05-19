# 41079 - Image Processing for Plant & Fruit Disease Detection

In this project we use convolutional neural networks (CNN) as classifiers for plant leaf diseases in Potato, Tomato, and Strawberry, and a Flask-based web app for inference and itneraction.

## Gaia: Directory Contents:w

- [Features](#features)
- [Setup](#setup)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Web App](#web-app)
  - [Data Preparation](#data-preparation)
  - [Training & Notebooks](#training--notebooks)
  - [Scripts](#scripts)
- [License](#license)

## Features
- Sequential + pretrained CNN training and Python notebooks
- Flask app interfacing real-time disease prediction
- Dataset utilities for combining and splitting classes
- Data storage including CNN model final + checkpointing

## Setup


1. Clone the repository and navigate into it:


   ```bash
   git clone <repo-url> 41079-research-project
   cd 41079-research-project
   ```

2. Create and activate a Python 3.10 virtual environment:

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate        #for macOS/Linux
   .\venv\Scripts\activate.ps1     #or Windows (incl. PS)
   ```

3. Install Python dependencies:


   ```bash
   pip install -r requirements.txt
   ```

## Datasets

This repository does not include large datasets. Download and organize them under `data/`:


1. [Potato Diseases Dataset](https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets)  
3. [Tomato Dataset](https://www.kaggle.com/datasets/ashishmotwani/tomato)  
4. [Strawberry Dataset](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset)

Each crop folder contain `train/`, `valid/`, and `test/` subfolders. Check [Dataset Preparation](#data-preparation) for final details.


## Usage
### Web App


Setup Flask server for the real-time plant processing:


```bash
cd app
export FLASK_APP=app.py     #for macOS/Linux
set FLASK_APP=app.py        #or Windows (incl. PS)
flask run
```
Visit http://127.0.0.1:5000 locally to upload plant/fruit images (strawberry, potato, tomato) for model predictions & probabilities


### Data Preparation


- `scripts/combine.py` and `scripts/combine_the_potatoes.py`: merges the potato folders into one directory (note that I have excluded the additional potato in README the other one can be found [here](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset))
- `scripts/split.py`: splits the tomato train & valid dataset into test equivocal test set

Example:


```bash
python scripts/split.py --src data/potato_combined --dst data/potato --ratios 0.7 0.15 0.15
```
```python
# On mac, use `Python3`
```

### Training & Notebooks


Run Jupyter notebooks to train and evaluate models:


```bash
jupyter lab notebooks/
```

All notebooks are:

- `notebooks/potato_cnn.ipynb`
- `notebooks/tomato_cnn.ipynb`
- `notebooks/strawberry_cnn.ipynb`

### Scripts


- `scripts/potato.py`: custom dataset loader for potatoes.
- `scripts/combine.py`: combine class folders.
- `scripts/split.py`: create train/valid/test splits.

## License
MIT LICENCE IN USE