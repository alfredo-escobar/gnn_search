import numpy as np
import os
import shutil
import json

dataset = "UNIQLO"
dataset_size = 1425
save_folder = "C:/Users/Alfredo/Desktop/new_test_sets/{}".format(dataset)
k = 10

def get_100_randoms():
    random_integers = np.random.randint(0, dataset_size, k)
    random_integers = np.sort(random_integers)

    visual_embeddings_catalog = "./catalogues/{}/ssearch/visual_embeddings_catalog.txt".format(dataset)

    with open(visual_embeddings_catalog, 'r') as file_ve:
        lines_ve = file_ve.read().splitlines()
    
    test_files = [lines_ve[i] for i in random_integers]

    count = 0

    for filename in test_files:
        source_path = "./catalogues/{}/train/{}".format(dataset, filename)
        destination_path = os.path.join(save_folder, "test", filename)
        
        try:
            shutil.copyfile(source_path, destination_path)
            count += 1
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return
        except IOError as e:
            print(f"Error: {e}")
            return
    
    if count == k:
        np.save(os.path.join(save_folder, "ssearch/test_set_integers.npy"), random_integers)

        with open('./similar_explorer/test_set_integers.json', 'w') as json_file:
            json.dump(random_integers.tolist(), json_file)

def sort_npy_file(input_file, output_file):
    # Load the .npy file
    data = np.load(input_file)

    # Ensure it's a 1D array
    if len(data.shape) != 1:
        raise ValueError("Input file does not contain a 1D array.")

    # Sort the array in ascending order
    sorted_data = np.sort(data)

    # Save the sorted array to a new .npy file
    np.save(output_file, sorted_data)

def npy_to_json(input_file):
    random_integers = np.load(input_file)
    with open('./similar_explorer/test_set_integers.json', 'w') as json_file:
        json.dump(random_integers.tolist(), json_file)


#get_100_randoms()
#sort_npy_file("ssearch/test_set_integers.npy", "test_set_integers_2.npy")
npy_to_json("./catalogues/UNIQLO/ssearch/test_set_integers.npy")