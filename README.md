# 🎬 Movie Recommendation System

This project builds a **Movie Recommendation System** using **TF-IDF Vectorization** and **Cosine Similarity**. It recommends similar movies based on features like genres, keywords, tagline, cast, and director.

## 📂 Dataset
The dataset used is a CSV file (`movies.csv`) containing **4,803 movies** with **24 attributes**, including:
- `genres`
- `keywords`
- `tagline`
- `cast`
- `director`
- and many more...

## 🛠️ Libraries Used
- `numpy`
- `pandas`
- `difflib`
- `sklearn` (`TfidfVectorizer`, `cosine_similarity`)
- `pandas_profiling` (for EDA)

## 🚀 How It Works

### 1. Importing Required Libraries
```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas_profiling as pf
```

### 2. Loading the Dataset
```python
movies_data = pd.read_csv("movies.csv")
movies_data.head()
Dataset Shape:
(4803, 24)
```
### 3. Exploratory Data Analysis (EDA)
```python
Generate an automatic profiling report:
pf.ProfileReport(movies_data)
Check for missing values:
movies_data.isna().sum()
```
### 4. Feature Selection
```python
We select the following features for the recommendation system:
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
Fill missing values with an empty string:
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
```

### 5. Combining Selected Features
```python
Combine the features into a single string:
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
```

### 6. Vectorizing the Features
```python
Convert text to feature vectors using TF-IDF:
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
Shape of the feature matrix:
(4803, 17318)
```

### 7. Finding Similar Movies
```pythonWe use Cosine Similarity to measure how similar movies are based on their feature vectors:
similarity = cosine_similarity(feature_vectors)

```

### 📌 How to Use
Clone the repository:


git clone https://github.com/PiyushJaiswall/Movie-Recommendation-System.git
Install the required libraries:


pip install -r requirements.txt
Run the script in your Python environment (Jupyter Notebook or any IDE).

💡 Future Improvements
Integrate a web interface with Streamlit or Flask.
Use additional NLP techniques to enhance feature extraction.
Optimize performance for large datasets.
🧑‍💻 Author
Piyush Jaiswal
