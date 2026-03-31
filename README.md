# CS 1851: Team ___ Final Project
**Authors:** Matt Prenovitz, Vanessa Alexander, and Veronica Figueroa

For this project's midterm submission, we created a Random Forest model to process the tabular data, a fully-connected CNN model to process the image data, and then implemented late fusion to combine predictions from both modalities into a final classification.

## File Descriptions:
**cnn_pipeline.py:** Python script that implements the CNN and FCN model pipelines for the image data with an added evaluate function and custom ImageDataset class for the multi modal late fusion pipeline in multi_modal.ipynb. 

**cnn_val_predictions.csv:** CSV file of cnn validation metrics with shape (700, 10). There are 700 validation samples and 10 features:

1. Unique sample ID  

2. true label 

3. cnn model class prediction 

4-10. cnn model probability for each class (0-6).

**collab_cnn_pipeline.ipynb:**
This contains the full CNN and FCN pipeline for processing image data, including simple graphs to visualize change in loss overtime. This is the most up to date file for running just the CNN and FCN portion and should be used over the cnn_pipeline.py file. To run, upload the file to Colab, and ensure the data is uploaded to collab in a directory called: final_data. After that, you can run all on the notebook. 

**fcn_val_predictions.csv:** CSV file of fcn validation metrics with shape (700, 10). There are 700 validation samples and 10 features:

1. Unique sample ID  

2. true label 

3. fcn model class prediction 

4-10. fcn model probability for each class (0-6). 

**multi_modal.ipynb:** Google Colab file that performs late fusion with the pipelines from the random_forest_pipeline.py and the cnn_pipeline.py files. Generates the validation predictions csv files that are used for the two late fusion techniques: average fusion and logistic regression with late fusion features. 

**random_forest_pipeline.py:** Python script that implements the Random Forest model pipeline for the tabular data.

**random_forest_pipeline_results.ipynb:** Google Colab file that contains the script from random_forest_pipeline.py and the associated outputs.

**rf_val_predictions.csv:** CSV file of random forest validation metrics with shape (700, 10). There are 700 validation samples and 10 features:

1. Unique sample ID  

2. true label 

3. rf model class prediction 

4-10. rf model probability for each class (0-6).


## Reasoning and Next Steps
For the tabular data model, we decided to start with **Random Forest** because it typically performs well on large datasets, is robust against overfitting due to its ensemble learning, and allows straightforward hyperparameter tuning via grid search. Should this model not perform too well, however, our next step could be to explore a **Gradient Boosting Machine** model, since it can often better handle class imbalance and can capture more complex feature interactions due to its sequential training and boosting learning process.

For the image data model, we decided to create a **fully-connected CNN** because it is well-equipped to handle image data, provides a good baseline for understanding full model behavior, and we do not currently have any prior assumptions that could dictate partial connections. Should this model not perform too well, we could switch to a **partially-connected CNN** that could result in faster training, potentially better generalization performance, and exploit any prior knowledge we extract from the data.

For the fusion concatenation, we decided to implement **late fusion** since this allowed us to train the tabular and image data models independently, which works well as a baseline approach. Based on performance, should we decide that we want to learn cross-modal interactions, we could implement **intermediate fusion**.
