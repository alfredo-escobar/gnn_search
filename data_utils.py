from unicodedata import name
import pandas as pd
import os
import numpy as np
import csv
import umap

def prepare_dataset(dataset, features, filenames, use_umap=False, n_components=None, seed=42):
    data = []
    text_data = []
    text_model = "roberta"
    #answers_df = pd.read_pickle("./catalogues/{}/data/{}/answers_dataset_vector.pkl".format(dataset, text_model))
    answers_df = pd.read_pickle("./catalogues/{}/data/{}/questions_dataset_vector.pkl".format(dataset, text_model))

    for i, embedding in enumerate(features):
        name_and_productid = os.path.splitext(os.path.basename(filenames[i]))[0]
        #print("Name_and_productid: {}".format(name_and_productid))
        try:
            desc_vector = answers_df.loc[answers_df['name_and_productid'] == name_and_productid, 'descVector'].iloc[0]
        except:
            print("Error with product {}. Omitting...".format(name_and_productid))
            continue
        data.append(np.array(embedding))
        text_data.append(np.array(desc_vector))

    if use_umap:
        reducer = umap.UMAP(n_components=n_components, random_state=seed)
        reducer.fit(text_data)
        text_data = reducer.transform(text_data)
        np.save("./pytorch_labels/{}/UMAP/{}-dim/visual_embeddings.npy".format(dataset, n_components), np.asarray(data))
        np.save("./pytorch_labels/{}/UMAP/{}-dim/text_embeddings.npy".format(dataset, n_components), np.asarray(text_data))

    else:
        #np.save("./pytorch_labels/{}/visual_embeddings.npy".format(dataset), np.asarray(data), delimiter=",")
        #np.save("./pytorch_labels/{}/text_embeddings.npy".format(dataset), np.asarray(text_data), delimiter=",")
        np.save("./pytorch_labels/{}/visual_embeddings.npy".format(dataset), np.asarray(data))
        np.save("./pytorch_labels/{}/text_embeddings.npy".format(dataset), np.asarray(text_data))


def delete_bad_products(path, new_path):
    df = pd.read_excel(path)
    new_df = df[~df['Title'].str.contains('/')]
    new_df.to_excel(new_path)

def find_products_without_images(data_path="./Datasets/Catalogo_Pepeganga/"):
    files = [os.path.splitext(filename)[0] for filename in os.listdir(data_path + "images/")]
    df = pd.read_excel(data_path + "data/answers_dataset_old.xlsx")
    for _, row in df.iterrows():
        title = row["Title"]
        productid = row["ProductId"]
        s_filename = "{}_{}".format(title, productid)
        if s_filename not in files:
            df.drop(_, inplace=True)
    df.to_excel(data_path + "data/answers_dataset.xlsx")

#delete_bad_products(path="./Datasets/Catalogo_Pepeganga/data/answers_dataset.xlsx", new_path="./Datasets/Catalogo_Pepeganga/data/answers_dataset2.xlsx")
#find_products_without_images()
