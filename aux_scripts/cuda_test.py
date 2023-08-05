import sys
import pandas as pd
import sklearn as sk
import tensorflow as tf
import torch


print(f"Tensor Flow Version: {tf.version}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


print(f"Torch Version: {torch.version}")
print(f"Torch GPU: {torch.cuda.is_available()}")
print(f"Torch GPU Name: {torch.cuda.get_device_name()}")

# print()
# print(f"Keras Version: {tf.keras.version}")
# print(f"Python {sys.version}")
# print(f"Pandas {pd.version}")
# print(f"Scikit-Learn {sk.version}")
