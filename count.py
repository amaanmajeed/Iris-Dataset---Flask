import numpy as np
import pandas as pd

# Load the iris dataset
df = pd.read_csv('iris.data', header=None)

# Convert the dataframe to a NumPy array
data = np.array(df.iloc[:, :-1])

# Find the minimum and maximum values of each column
min_values = np.min(data, axis=0)
max_values = np.max(data, axis=0)

# Print the results
for i in range(len(min_values)):
    print("Column {}: min={}, max={}".format(i+1, min_values[i], max_values[i]))

