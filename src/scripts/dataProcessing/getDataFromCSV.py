import csv
import urllib.request
from urllib.error import HTTPError
from keras.preprocessing.image import img_to_array, load_img
import os


def clear_images_from_folder_by_dimensions(x, y, folder):
    # get all images
    only_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    number_deleted_images = 0

    for i in range(len(only_files)):
        img = load_img(folder + "/" + only_files[i])
        img_np = img_to_array(img)
        if img_np.shape[0] == x and img_np.shape[1] == y:
            os.remove(folder + "/" + only_files[i])
            print("Delete Image " + only_files[i])
            number_deleted_images += 1

    print("Number deleted Images: " + number_deleted_images)


def scrape_momuk_images_from_website(directory):
    with open('momok_kunstwerke.csv') as csv_file:
        line_counter = 0
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if line_counter != 0:
                obj_id = row[0]
                obj_url = row[19]
                filename = directory + obj_id + ".jpeg"
                try:
                    urllib.request.urlretrieve(obj_url, filename)
                except FileNotFoundError as err:
                    print(err)  # something wrong with local path
                except HTTPError as err:
                    print(err)  # something wrong with url

            line_counter += 1
