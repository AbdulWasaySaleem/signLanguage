import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import os

# Load the trained Random Forest model
with open('trained_model.pickle', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']  # The trained Random Forest

# Create a directory to store the plots of each tree
if not os.path.exists('rf_trees'):
    os.makedirs('rf_trees')

# Loop over all estimators in the Random Forest
for idx, specific_tree in enumerate(model.estimators_):
    # Plot the individual tree
    plt.figure(figsize=(12, 8))
    tree.plot_tree(
        specific_tree,
        feature_names=None,  # Feature names if you have them
        class_names=None,  # Class names if you have them
        filled=True,  # Color coding the tree for easier visualization
        rounded=True,  # Rounded corners for better aesthetics
        precision=2,  # Decimal precision for numbers
    )
    plt.title(f"Decision Tree #{idx + 1}")  # Title indicating which tree it is
    plt.savefig(f'rf_trees/decision_tree_{idx + 1}.png')  # Save the plot as a PNG
    plt.close()  # Close the plot to free memory
