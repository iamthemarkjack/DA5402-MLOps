# Assignment-09 EP21B030 - Sentiment Analysis with Apache Spark

This assignment leverages PySpark for scalable sentiment analysis and evaluation on Amazon Gourmet Foods reviews. It uses transformer-based embeddings for classification and produces key performance metrics and visualizations.

### `Task_1.py`

**Key Features:**
- Reads and parses reviews from `./data/Gourmet_Foods.txt`.
- Extracts relevant fields: product ID, rating, and review text.
- Applies a transformer model (`sentence-transformers/all-MiniLM-L6-v2`) to generate sentence embeddings.
- Uses PySpark to parallelize sentiment classification for faster inference.
- Saves the results to `./outputs/task1_predictions.txt` with the format:
  
  ```
  product_id <TAB> rating <TAB> review_text <TAB> sentiment
  ```

- Logs progress and errors in `./logs/task1.log`.

### `Task_2.py`

**Key Features:**
- Loads model predictions from `./outputs/task1_predictions.txt`.
- Computes precision and recall based on a thresholded star rating.
- Constructs a confusion matrix and visualizes it using Seaborn and Matplotlib.
- Saves the plot to `./outputs/task2_confusion_matrix.png`.
- Records execution steps and issues in `./logs/task2.log`.

##  Running the Pipeline

### 1. Prepare Input Data

Ensure the reviews file is placed in the correct location:

```
./data/Gourmet_Foods.txt
```

### 2. Run Sentiment Analysis

Execute the following command to classify sentiments in parallel:

```bash
python Task_1.py
```

### 3. Run Evaluation & Plotting

Generate performance metrics and the confusion matrix visualization:

```bash
python Task_2.py
```