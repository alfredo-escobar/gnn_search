import datasets.data as data
import numpy as np
import os
import json
import pandas as pd


def get_product_and_category(r_filenames, dataframe):
    df = dataframe
    products = []

    for i, file in enumerate(r_filenames):
        base = os.path.basename(file)
        filename = os.path.splitext(base)[0]
        name_and_productid = filename.rsplit('_', 1)

        try:
            categories = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
        except: 
            try:
                categories = df.loc[(df['Title'] == name_and_productid[0]) & (str(df['ProductId']) == name_and_productid[1]), ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            except:
                categories = df.loc[df['Title'] == name_and_productid[0], ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
        if i == 0:
            base_categories = categories
        else:
            file_info = [filename, categories[0], categories[1], categories[2]]
            products.append(file_info)

    return base_categories, products


def count(products):
    categories = {}
    for product in products:
        if product[1] in categories:
            categories[product[1]] += 1
        else:
            categories[product[1]] = 1
    
    return categories
    


def explore(dataset, prod_a, prod_b, prod_c):

    only_get_gc = False

    data_path = "./catalogues/{}/data/".format(dataset)
    df = pd.read_excel(data_path + "categoryProductsES_EN.xlsx")

    model_descriptions = {
        "resnet_50"       : "según el \\textbf{baseline} ResNet-50 (sin entrenamiento)",
        "cos_slineal"     : "tras 1 iteración de entrenamiento con \\textbf{matriz de similitud coseno y función \\textit{s\_linear}}",
        "cos_sprobs"      : "tras 1 iteración de entrenamiento con \\textbf{matriz de similitud coseno y función \\textit{s\_prob}}",
        "cossoft_slineal" : "tras 1 iteración de entrenamiento con \\textbf{matriz de similitud coseno con \\textit{softmax} y función \\textit{s\_linear}}",
        "cossoft_sprobs"  : "tras 1 iteración de entrenamiento con \\textbf{matriz de similitud coseno con \\textit{softmax} y función \\textit{s\_prob}}"
        }

    all_r_filenames = {}
    for model in model_descriptions.keys():
        with open('./jsons/{}/{}.json'.format(model, dataset), "r") as json_file:
            r_filenames = json.load(json_file)
        all_r_filenames[model] = r_filenames

    for prod_idx in [prod_a, prod_b, prod_c]:

        for model, r_filenames in all_r_filenames.items():

            base_category, products = get_product_and_category(r_filenames[prod_idx], df)
            print(f"\nDataset: {dataset}, Model: {model}, Product index: {prod_idx}")

            if only_get_gc:
                print("\\\ \hspace*{1cm} \\textbf{Categoría global:} ", base_category[0])
                print("\n----------\n")
                break

            categories = count(products)

            #caption = "    \\caption{Recuperación de los 20 productos más similares a la \\textit{query} "
            #caption += model_descriptions[model]
            #caption += ". Arriba a la izquierda se presenta la imagen de consulta, y el orden de los resultados recuperados es de izquierda a derecha, y de arriba hacia abajo."

            caption = ""
            cat_idx = 1
            for category, quantity in categories.items():

                if cat_idx == 1:
                    caption += f" De estos, el {int(quantity * 100 / len(products))}\% corresponde a la categoría ``{category}''"

                elif cat_idx == len(categories):
                    caption += f" y el {int(quantity * 100 / len(products))}\% corresponde a ``{category}''"

                else:
                    caption += f", el {int(quantity * 100 / len(products))}\% corresponde a ``{category}''"

                cat_idx += 1
            
            caption += "."
            #caption += "}"
            
            print(caption)
            print("\n----------\n")


explore("UNIQLO"   , 34, 18, 98)
explore("IKEA"     ,  3, 28, 91)
explore("Pepeganga",  0, 78, 92)