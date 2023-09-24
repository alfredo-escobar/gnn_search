import datasets.data as data
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import utils.imgproc as imgproc
import skimage.transform as trans
import skimage.io as io
import json


def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")
    except OSError as e:
        print(f"Error creating folder '{folder_path}': {e}")


def draw_result(filenames, write_data=False, similarity=None, distance=None):
    rows = int(np.ceil((len(filenames) - 1) / 10)) + 1

    w_i = 100
    h_i = 100

    w = w_i * 10
    h = h_i * rows

    image_r = np.zeros((h,w,3), dtype = np.uint8) + 255
    x = 0
    y = 0
    for i, filename in enumerate(filenames) :
        if i == 0:
            pos = 0
        else:
            pos = ((i + 9) * w_i)
        x = pos % w
        #y = np.int(np.floor(pos / w)) * h_i
        y = int(np.floor(pos / w)) * h_i
        image = data.read_image(filename, 3)
        
        if write_data:
            ### Add text with the product id
            try:
                base = os.path.basename(filename)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                font = ImageFont.truetype("arial.ttf", 30)
                PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
                draw = ImageDraw.Draw(PIL_image)
                if (similarity is None and distance is None) or (i == 0):
                    draw.text((0, 0), "id: {}".format(name_and_productid[1]), font=font, fill='rgb(0, 0, 0)')
                elif similarity is not None:
                    draw.text((0, 0), "id: {} / sim: {}".format(name_and_productid[1], round(similarity[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                elif distance is not None:
                    draw.text((0, 0), "id: {} / dist: {}".format(name_and_productid[1], round(distance[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
            except:
                #print("Could not write id for product.")
                pass
            image = np.array(PIL_image)

        image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
        image_r[y:y+h_i, x : x +  w_i, :] = image              
    return image_r


dataset = input("Enter dataset name: ")
current_iteration = input("Enter iter number: ")

with open('./catalogues/{}/results/search_results/iter_{}.json'.format(dataset, current_iteration), "r") as json_file:
    all_r_filenames = json.load(json_file)

folder_name = "./catalogues/{}/results/search_results/iter_{}".format(dataset, current_iteration)
create_folder(folder_name)

for r_filenames in all_r_filenames:
    
    image_r = draw_result(r_filenames)

    fquery = r_filenames[0] # Full path
    fquery = os.path.basename(fquery) # Filename with extension
    output_name = os.path.splitext(fquery)[0]  + '.png' # Filename with extension replaced to png
    output_name = os.path.join(folder_name, output_name)
    io.imsave(output_name, image_r)
    print('result saved at {}'.format(output_name))