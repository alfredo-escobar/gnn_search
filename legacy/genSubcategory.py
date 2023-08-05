import csv

def genSubcategory(dataset):
    # Open the CSV file for reading
    with open('./catalogues/' + dataset + '/test.csv', 'r') as file:
        reader = csv.reader(file, delimiter=';')

        # Create a new list to hold the rows with the added column
        rows_with_column = []

        next(reader)

        # Loop through each row in the CSV file
        for row in reader:

            # Extract the list of four elements from column F
            f_list = row[5].split('/')

            # Extract the third element of the list and save it in a new column
            row.append(f_list[2])

            # Add the modified row to the list of rows with the added column
            rows_with_column.append(row)

    # Open the CSV file for writing
    with open('./catalogues/' + dataset + '/data/questions_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')

        writer.writerow(["Unnamed: 0","Url","ImageUrl","ProductId","Title","CategoryTree","ProductDescriptionEN","ProductDescriptionEN2","ProductDescriptionEN3","GlobalCategoryEN","SubCategory"])

        # Write the rows with the added column to the new file
        for row in rows_with_column:
            writer.writerow(row)