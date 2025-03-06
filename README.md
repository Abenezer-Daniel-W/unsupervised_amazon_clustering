# Amazon Forest Clustering Project

This project performs unsupervised clustering on images of the Amazon forest using KMeans and Gaussian Mixture Models (GMM). The images are divided into chunks, and the mean RGB values of each chunk are calculated for clustering.

## Requirements

- Python 3.x
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib

Install the required packages using:

```sh
pip install -r requirements.txt
```

## Project Structure

- forest_full.ipynb: Jupyter notebook for loading, processing, and clustering images.
- web_interface.py: Streamlit app for visualizing clustering results.
- README.md: Project documentation.
- requirements.txt: List of required Python packages.

## Usage

### Running the Jupyter Notebook

1. Open forest_full.ipynb in Jupyter Notebook or JupyterLab.
2. Execute the cells to load, process, and cluster the images.

### Running the Streamlit App

1. Ensure that the training and test images are placed in the appropriate directories:

   - Training images: `Amazon Forest Dataset/Training/images`
   - Test images: `Amazon Forest Dataset/Test/`

2. Run the Streamlit app:

```sh
streamlit run web_interface.py
```
