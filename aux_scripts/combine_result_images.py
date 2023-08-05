from PIL import Image
from ntpath import join
import os

def combine_images_vertically(images, spacing):
    total_height = sum(img.size[1] for img in images) + spacing * (len(images) - 1)
    max_width = max(img.size[0] for img in images)

    combined_image = Image.new('RGB', (max_width, total_height), color='white')

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.size[1] + spacing

    return combined_image

def main():
    folder_prefix = './catalogues/UNIQLO/results/iter_'
    output_folder = './catalogues/UNIQLO/results/output_path/'

    filenames_path = "./catalogues/UNIQLO/results/iter_0"
    filenames =  [f for f in os.listdir(filenames_path) if os.path.isfile(join(filenames_path, f))]

    for filename in filenames:
        images = []
        for j in [0,1,6,11,16,21,26,31]:
            image_path = f"{folder_prefix}{j}/{filename}"
            img = Image.open(image_path)
            images.append(img.crop((0, 0, img.width, 200)))  # Crop the upper 200px section

        combined_img = combine_images_vertically(images, spacing=50)
        output_path = f"{output_folder}{filename}.png"
        combined_img.save(output_path)

if __name__ == "__main__":
    main()
