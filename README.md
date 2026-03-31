# CS 1851: Team ___ Final Project

For this project's midterm submission, we created a Random Forest model to process the tabular data, a fully-connected CNN model to process the image data, and then implemented late fusion to combine predictions from both modalities into a final classification.

## File Descriptions:
**cnn_pipeline.py:**

**cnn_val_predictions.csv:**

**collab_cnn_pipeline.ipynb:**

**collab_cnn_pipeline.ipynb:**

**fcn_val_predictions.csv:**

**multi_modal.ipynb:**

**preprocess.py:**

**random_forest_pipeline.py:** Python script that implements the Random Forest model pipeline for the tabular data.

**random_forest_pipeline_results.ipynb:** Google Colab file that contains the script from random_forest_pipeline.py and the associated outputs.

**rf_val_predictions.csv:**


## Reasoning and Next Steps
For the tabular data model, we decided to start with **Random Forest** because it typically performs well on large datasets, is robust against overfitting due to its ensemble learning, and allows straightforward hyperparameter tuning via grid search. Should this model not perform too well, however, our next step could be to explore a **Gradient Boosting Machine** model, since it can often better handle class imbalance and can capture more complex feature interactions due to its sequential training and boosting learning process.

For the image data model, we decided to create a **fully-connected CNN** because it is well-equipped to handle image data, provides a good baseline for understanding full model behavior, and we do not currently have any prior assumptions that could dictate partial connections. Should this model not perform too well, we could switch to a **partially-connected CNN** that could result in faster training, potentially better generalization performance, and exploit any prior knowledge we extract from the data.

For the fusion concatenation, we decided to implement late fusion since this allowed us to train the tabular and image data models independently, which works well as a baseline approach. Based on performance, should we decide that we want to learn cross-modal interactions, we could implement **intermediate fusion**.
