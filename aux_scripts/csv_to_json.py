import csv
import json

def csv_to_json(csv_file, json_file):
    data_dict = {}
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row_number, row in enumerate(csvreader):
            if len(row) >= 9:
                #values_list = [row[4] + "_" + row[3] + ".jpg", row[4], row[5], row[6], row[7], row[8]]
                values_list = [row[2], row[4], row[5], row[6], row[7], row[8]]
                data_dict[row_number] = values_list

    with open(json_file, 'w') as jsonfile:
        json.dump(data_dict, jsonfile)

csv_to_json('./catalogues/UNIQLO/train_search.csv', './similar_explorer/catalogue.json')
