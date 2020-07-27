# Generation of artifical artwork

This is a toolkit for generation of artificial artworks using a generative adversial network.
It includes a scraper for image collection from online sources, a training script for a GAN and a GUI for generating and editing images.

## Prerequisites

To execute all scripts, a python environment with tensorflow is needed. Using gpu support for tensorflow is highly recommended.

Additionally used Python packages:
* Keras
* LMDB
* Scrapy
* cv2
* selenium
* kivy

### Scraping data

The scraper downloads images of a specific category and type from an open image database. Because the website uses JavaScript, selenium is needed besides scrapy to emulate a browser. So the workload is higher than simple HTML scraping.
To start the spider and downloading artwork from a website, the following command must be executed:

`scrapy crawl wgaSpider -a category=painting -a typeof=landscape -a folder=./mydata/`

With the parameter category and landscape can the images be defined. Additional information is defined in the code.

### Data processing

The GAN uses LMDB as a database for the images while training. So the downloaded images must be preprocessed and inserted in a lmdb file. This is done using the preprocessing script. Before executing dataProcessing.py the database size, image resolution and data location must be adjusted in the main method.

```
lmdb = LMDBase('data/lmdb', 200000 * 7500 * 10)
lmdb.store_images_from_folder('data/landscape_painting', 128, 128, 100)
```

### Training the GAN

By executing the gan_mult_gpus.py script, the GAN will be trained. Before training, the parameter in the main method must be adjusted. The resolution, number and location of the images need to be changed. Changing the number of epochs, batches and latent_dim size can influence the result of the training. If the workload is to high reduce image size, number epochs and batch size.

For the training is a workstation with a powerful gpu recommended. The script can use theoretically multiple gpus, but this is not tested. Depending on computing power the trainig can last a multiple hours to days.

### Generate images using GUl

After training the GAN new images can be generated using the created models/weights and the GUI. Before starting the GUI the path to the models and weights must be adjusted in the main_ui.py script. With the GI images can now be created, edited using multiple filters and be saved.
