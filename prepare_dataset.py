""" 
Current requirements:
- "embeddings/CLIP/visual_embeddings.npy"
- "embeddings/ResNet/visual_embeddings.npy"
- "embeddings/RoBERTa/text_embeddings.npy"
- "ssearch/visual_embeddings_catalog.txt"
- "ssearch/text_embeddings_catalog.txt"
"""

import os
import numpy as np
import pandas as pd
from openpyxl import Workbook


def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    except OSError as e:
        print(f"Error creating folder '{folder_path}': {e}")


def process_csv_to_xlsx(df, output_file):
    # Function to extract the third element from a '/' separated string
    def get_subcategory(category_tree):
        categories = category_tree.split("/")
        return categories[2] if len(categories) >= 3 else ""

    # Add a new column "SubCategory" and populate it with the values from "CategoryTree"
    df["SubCategory"] = df["CategoryTree"].apply(get_subcategory)

    # Create a new Excel workbook and add a worksheet
    workbook = Workbook()
    worksheet = workbook.active

    # Convert the DataFrame to a list of lists and add it to the worksheet
    data = [df.columns.tolist()] + df.values.tolist()
    for row in data:
        worksheet.append(row)

    # Save the workbook to the output file
    workbook.save(output_file)


def generate_XLSX(dataset):
    # Process train.csv and save to answers_dataset.xlsx
    
    input_train_csv_dataset = pd.read_csv("./catalogues/{}/train.csv".format(dataset), sep=";")
    output_answers_xlsx_file = "./catalogues/{}/data/answers_dataset.xlsx".format(dataset)
    process_csv_to_xlsx(input_train_csv_dataset, output_answers_xlsx_file)

    # Process test.csv and save to questions_dataset.xlsx
    input_test_csv_dataset = pd.read_csv("./catalogues/{}/test.csv".format(dataset), sep=";")
    output_questions_xlsx_file = "./catalogues/{}/data/questions_dataset.xlsx".format(dataset)
    process_csv_to_xlsx(input_test_csv_dataset, output_questions_xlsx_file)

    # Create a combined dataset using the two CSV files
    combined_dataset = pd.concat([
        input_train_csv_dataset,
        input_test_csv_dataset
    ], ignore_index=True)

    # Process the combined dataset and save to categoryProductsES_EN.xlsx
    output_combined_xlsx_file = "./catalogues/{}/data/categoryProductsES_EN.xlsx".format(dataset)
    process_csv_to_xlsx(combined_dataset, output_combined_xlsx_file)


def generate_NP(dataset, model = "ResNet"):
    # Load the data from the .npy file
    a = np.load("./catalogues/{}/embeddings/{}/visual_embeddings.npy".format(dataset, model))

    # Save data to a new .np file
    a.astype(np.float32).tofile("./catalogues/{}/ssearch/features.np".format(dataset))

    # Save the shape of the array to a new .np file
    np.asarray(a.shape).astype(np.int32).tofile('./catalogues/{}/ssearch/features_shape.np'.format(dataset))


if __name__ == "__main__":

    dataset = "Homy"
    model = "ResNet"

    create_folder("./catalogues/{}/data".format(dataset))
    create_folder("./catalogues/{}/data/word2vec".format(dataset))
    create_folder("./catalogues/{}/results".format(dataset))
    create_folder("./catalogues/{}/ssearch".format(dataset))
    
    generate_XLSX(dataset)
    generate_NP(dataset, model)