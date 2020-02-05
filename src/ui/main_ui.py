from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Rectangle, Color
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, BooleanProperty
from kivy.factory import Factory

from keras.preprocessing import image as keraimg
from keras.models import model_from_json

import json
import numpy as np
import easygui
import os
from PIL import Image
import cv2


class DrawingWidget(BoxLayout):
    image = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.clearcolor = (1, 1, 1, 1)
        self.bind(pos=self.update)
        self.bind(size=self.update)
        self.update()

    # updating image when window size/pos changes
    def update(self, *args):
        # redraw image if image exist
        if (self.image is None) is False:
            # self.on_image(None, None)
            pass

    def paint_image(self):
        if (self.image is None) is False:
            # clear image
            self.canvas.clear()
            # TODO check for window boundries and scale image if possible
            img = self.image
            x_start = int(self.size[1] / 2 - img.shape[1] / 2)
            y_start = int(self.size[0] / 2 - img.shape[0] / 2)
            i_max = img.shape[1]-1

            for i in range(img.shape[1]):
                for y in range(img.shape[0]):
                    with self.canvas:
                        # create color pixel
                        Color(img[i_max-i][y][0] / 255, img[i_max-i][y][2] / 255, img[i_max-i][y][1] / 255, 1)
                        Rectangle(size=(1, 1), pos=(y+y_start, i+x_start))

    # drawing new image, when data changes
    def on_image(self, instance, value):
        self.paint_image()


class Interface(GridLayout):
    image = ObjectProperty(None)
    last_image = ObjectProperty(None)
    model = ObjectProperty(None)

    model_loaded = BooleanProperty(False)
    image_generated = BooleanProperty(False)
    undo_possible = BooleanProperty(False)

    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)

    _popup = None

    def on_image(self, instance, value):
        # self.drawing_widget.image = self.image
        pass

    def save_progress(self):
        if (self.image is None) is False:
            self.undo_possible = True
            self.last_image = self.image

    def undo_step(self, drawing_widget):
        if (self.last_image is None) is False and self.undo_possible is True:
            self.undo_possible = False
            self.image = self.last_image
            drawing_widget.image = self.image

    def load_model(self):
        self.open_loading_popup("Loading model...")
        # path_model = easygui.fileopenbox(msg="Choose model file", title="gan model")
        # path_weights = easygui.fileopenbox(msg="Choose weights file", title="gan weights")
        path_model = "/home/steffen/PycharmProjects/neuronart/src/gan/landscape.v1.0/landscape1/generator_model.json"
        path_weights = "/home/steffen/PycharmProjects/neuronart/src/gan/landscape.v1.0/landscape1/generator_weights.h5"

        with open(path_model, 'r') as f:
            model_json = json.load(f)

        model = model_from_json(model_json)
        model.load_weights(path_weights)
        model.summary()
        self.model = model
        self.model_loaded = True
        self._popup.dismiss()

    def generate_img(self, drawing_widget):
        if self.model is None:
            print("No model loaded")
        else:
            self.save_progress()
            self.open_loading_popup("Generating image...")
            # generate some images
            num_images = 1
            latent_dim = 150
            random_latent_vectors = np.random.normal(size=(num_images, latent_dim))
            generated_images = self.model.predict(random_latent_vectors)

            # convert generated image to rgb image
            # TODO find smarter way to convert
            self.image = np.array(keraimg.array_to_img(generated_images[0] * 255., scale=False))
            drawing_widget.image = self.image

            # set image as generated
            self.image_generated = True

            self._popup.dismiss()

    def average_image(self, drawing_widget, size):
        self.save_progress()
        size = int(size)
        self.image = cv2.blur(self.image, (size, size))
        drawing_widget.image = self.image

    def median_image(self, drawing_widget, size):
        self.save_progress()
        size = int(size)
        self.image = cv2.medianBlur(self.image, size)
        drawing_widget.image = self.image

    def bilateral_filter_image(self, drawing_widget, pixel_diameter, sigma_color, sigma_space):
        self.save_progress()
        self.image = cv2.bilateralFilter(self.image, pixel_diameter, sigma_color, sigma_space)
        drawing_widget.image = self.image

    def gaussian_image(self, drawing_widget, size):
        if size % 2 == 1:
            self.save_progress()
            size = int(size)

            gaussian_filtered_image = cv2.GaussianBlur(self.image, (size, size), 0)
            print("Gaussian: " + str(gaussian_filtered_image))
            self.image = gaussian_filtered_image
            drawing_widget.image = self.image
        else:
            self._popup = Popup(title='Failed filtering', content=Label(text='Use only odd number for gaussin kernel'), size=(200,200))
            self._popup.open()

    def open_loading_popup(self, text):
        self._popup = Popup(title='Progressing Task', content=Label(text=text),
                            auto_dismiss=False, size_hint=(None, None), size=(200, 200))
        self._popup.open()

    def dismiss_popup(self):
        if (self._popup is None) is False:
            self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            self.text_input.text = stream.read()

        self.dismiss_popup()

    def save(self, path, filename):
        im = Image.fromarray(self.image)
        print("Path: ", path)
        print("Filename: ", filename)
        im.save(path + '/' + filename + ".jpeg")

        self.dismiss_popup()


class DrawingApp(App):

    def build(self):
        root_widget = Interface()
        return root_widget


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)
DrawingApp().run()
