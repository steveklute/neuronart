# -*- coding: utf-8 -*-
import scrapy
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
import urllib.request
from selenium.common.exceptions import NoSuchElementException
import time
import os
import errno
from selenium.webdriver.firefox.options import Options


class WgaspiderSpider(scrapy.Spider):
    """
    This class scrapes the wga website and download all images based on category and type.

    The path to geckodriver needs to be added as path environment variable.

    Should be executed with scrapy like:
    'scrapy crawl wgaSpider -a category=painting -a typeof=landscape -a folder=./mydata/' from the commandline
    in the root of the project.

    Keyword arguments:
    category -- the category of the artworks (default is any)
    typeof -- the type of the artworks (default is any)
    folder -- the folder, where the images will be saved (default is actual folder)
    site_load_delay -- after every click on the website the next operation must be delayed. Change this value depending
                        on computer and internet speed (default is 2.5)
    window_switch_delay -- after every switch to a new window the next operation must be delayed. Change this value
                            depending on computer and internet speed (default is 1)

    all categories: any, painting, sculpture, graphics, illumination, architecture, ceramics, furniture, glassware,
                    metalwork, mosaic, stained-glass, tapestry, others
    all types: any, religious, historical, mythological, landscape, portrait, still-life, interior, genre, study, others
    """
    name = 'wgaSpider'
    allowed_domains = ['www.wga.hu']
    start_urls = ['https://www.wga.hu/']

    def __init__(self, category='any', typeof='any', folder='', site_load_delay=2.5, window_switch_delay=1, *args, **kwargs):
        super(WgaspiderSpider, self).__init__(*args, **kwargs)

        # set parameters
        self.category = category
        self.typeof = typeof
        self.folder = folder
        self.site_load_delay = site_load_delay
        self.window_switch_delay = window_switch_delay

        # init folder
        try:
            os.makedirs(self.folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # init webdriver
        options = Options()
        options.headless = True
        self.driver = webdriver.Firefox(options=options)
        self.driver.minimize_window()

    def parse(self, response):
        """
        The parse method get automatically called by scapy and should no be used manuel.
        :param response: The http response of the actual scraped website
        :return: yields for scrapy
        """

        # open the next website
        self.driver.get(response.url)

        # image id will be used as filename
        actual_img_id = 0

        while True:
            # hit enter button, which is 'hidden' from bots, by manuel click
            img_over_enter_button = self.driver.find_element_by_xpath('/html/body/table/tbody/tr[5]/td[1]/img')
            try:
                ActionChains(self.driver).move_to_element_with_offset(img_over_enter_button, 35, 33).click().perform()
                # Wait for javscript to load in Selenium
                time.sleep(self.site_load_delay)

                # switch to new frame
                frame = self.driver.find_element_by_xpath('/html/frameset/frame[2]')
                self.driver.switch_to.frame(frame)

                try:
                    # click link to get to search engine
                    self.driver.find_element_by_link_text('Search Engine').click()

                    # Wait for javscript to load in Selenium
                    time.sleep(self.site_load_delay)

                    # fill search engine forms
                    select = Select(self.driver.find_element_by_name('form'))
                    select.select_by_visible_text(self.category)
                    select = Select(self.driver.find_element_by_name('type'))
                    select.select_by_visible_text(self.typeof)

                    # get search button and click it
                    search_engine_button = self.driver.find_element_by_xpath(
                                                                            '/html/body/form/center/p[4]/font/input[1]')
                    try:
                        search_engine_button.click()
                        # Wait for javscript to load in Selenium
                        time.sleep(self.site_load_delay)

                        # endless loop for iterating over all pages
                        while True:
                            # get all image boxes from actual page
                            image_boxes = self.driver.find_elements_by_xpath('/html/body/center[3]/table/tbody/tr')

                            # get next page button
                            next_page_button = self.driver.find_elements_by_xpath(
                                                                    '/html/body/center[4]/table/tbody/tr/td/p/a')[-1]

                            # for loop over image boxes on the current page
                            i = 2
                            while i < len(image_boxes):
                                # save main window
                                window_before = self.driver.window_handles[0]

                                # open popup window with image
                                preview_img = self.driver.find_element_by_xpath('/html/body/center[3]/table/tbody/tr['
                                                                                + str(i) + ']/td[1]/a/img')
                                preview_img.click()

                                # switch to popup window
                                self.driver.switch_to.window(self.driver.window_handles[1])
                                time.sleep(self.site_load_delay)

                                # switch to new frame in popup window
                                frame = self.driver.find_element_by_xpath('/html/frameset/frame[2]')
                                self.driver.switch_to.frame(frame)

                                # get and save the image
                                image = self.driver.find_element_by_xpath('/html/body/center/img')
                                src = image.get_attribute('src')
                                urllib.request.urlretrieve(src, self.folder + str(actual_img_id) + '.jpg')

                                # increment image id
                                actual_img_id += 1

                                # close popup window
                                self.driver.close()
                                time.sleep(self.window_switch_delay)

                                # switch to main window
                                self.driver.switch_to.window(window_before)

                                # increment counter for images on the current page
                                i += 1

                            # switch to next page
                            try:
                                next_page_button.click()
                            except NoSuchElementException as exception:
                                print("couldn't switch to next page: ", exception)
                                break
                    except NoSuchElementException as exception:
                        print("Failed to click search button: ", exception)
                        break
                except NoSuchElementException as exception:
                    print("Failed to click search engine link: ", exception)
                    break
            except NoSuchElementException as exception:
                print("couldn't click enter button: ", exception)
                break

        # close main window to end scraping
        self.driver.close()
