import pickle
import pandas as pd

# Sample dataset
data = {'Username': ['Alice', 'Bob', 'Charlie'], 'Password': ['alice123','alice456', 'alice789']}
df = pd.DataFrame(data)

# Save dataset to a binary file using pickle
with open('example.bin', 'wb') as file:
    pickle.dump(df, file)

# Load dataset from the binary file
with open('example.bin', 'rb') as file:
    loaded_df = pickle.load(file)

print(loaded_df)