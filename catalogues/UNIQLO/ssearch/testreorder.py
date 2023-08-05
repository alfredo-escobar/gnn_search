import numpy as np

ve_catalog = "./catalogues/UNIQLO/ssearch/catalog.txt"
te_catalog = "./catalogues/UNIQLO/ssearch/text_embeddings_catalog.txt"

# Read the contents of a.txt and b.txt
with open(ve_catalog, 'r') as file_ve, open(te_catalog, 'r') as file_te:
    lines_ve = file_ve.read().splitlines()
    lines_te = file_te.read().splitlines()

# Create a NumPy array c
indices = np.zeros(len(lines_ve), dtype=np.int32)

# Populate c with positions
for i, element in enumerate(lines_ve):
    indices[i] = lines_te.index(element[0:-4])

lines_te[indices]

print(indices)
