import streamlit as st
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import joblib

# Define the paths to the training and test images
test_path = "Amazon Forest Dataset/Test/"
train_path = "./Amazon Forest Dataset/Training/images"

# Load the pre-trained models
kmeans = joblib.load('models/kmeans_model.pkl')
gmm = joblib.load('models/gmm_model.pkl')

@st.cache_data
def get_random_image_names(path, num_images=7):
    image_names = os.listdir(path)
    return random.sample(image_names, num_images)

# Select 7 random images from the test path
random_names = get_random_image_names(test_path)

def predict_and_plot(image_name, model, model_name):
    img = Image.open(os.path.join(test_path, image_name))
    img_array = np.array(img, dtype=np.float32)
    pixels = img_array.reshape(-1, 3)
    predicted_labels = model.predict(pixels)
    predicted_image = predicted_labels.reshape(img_array.shape[:2])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_array / 255.0)
    ax[0].set_title('Original Image')
    ax[1].imshow(predicted_image, cmap='viridis')
    ax[1].set_title(f'Predicted Image ({model_name})')
    st.pyplot(fig)

# Streamlit interface
st.title("Amazon Forest Clustering Project")
st.write("""
This project performs unsupervised clustering on images of the Amazon forest using KMeans and Gaussian Mixture Models (GMM). 
The images are divided into chunks, and the mean RGB values of each chunk are calculated for clustering.
""")

st.header("Algorithms Used")
st.write("""
### KMeans
KMeans is a clustering algorithm that partitions the data into K clusters. Each data point belongs to the cluster with the nearest mean. The algorithm iteratively updates the cluster centers and assigns data points to the nearest cluster until convergence.

### Gaussian Mixture Model (GMM)
GMM is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions with unknown parameters. The algorithm uses the Expectation-Maximization (EM) algorithm to estimate the parameters of the Gaussian distributions and assign data points to the clusters.
""")

st.header("Clustering Results")
model_option = st.selectbox("Select Model", ("KMeans", "GMM"))

if model_option == "KMeans":
    model = kmeans
    model_name = "KMeans"
else:
    model = gmm
    model_name = "GMM"

for name in random_names:
    st.subheader(f"Image: {name}")
    predict_and_plot(name, model, model_name)

st.header("About Me")
st.write("""
**Abenezer Woldesenbet**  
[ad5643007@gmail.com](mailto:ad5643007@gmail.com) | [LinkedIn](https://linkedin.com/in/abenezer-woldesenbet/) | [GitHub](https://github.com/Abenezer-Daniel-W)
""")