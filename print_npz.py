import numpy as np

data = np.load("color_collection.npz", allow_pickle=True)

print(data.files)  # Lists keys
print(data["condition"])  # View content of a key