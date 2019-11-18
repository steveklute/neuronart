# -*- coding: utf-8 -*-
import scrapy
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
import urllib.request
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
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
    max_site_load_delay -- after every click on the website the next operation must be delayed. Change this value depending
                        on computer and internet speed (default is 2.5)

    all categories: any, painting, sculpture, graphics, illumination, architecture, ceramics, furniture, glassware,
                    metalwork, mosaic, stained-glass, tapestry, others
    all types: any, religious, historical, mythological, landscape, portrait, still-life, interior, genre, study, others
    """
    name = 'wgaSpider'
    allowed_domains = ['www.wga.hu']
    start_urls = ['https://www.wga.hu/']

    def __init__(self, category='any', typeof='any', folder='', max_site_load_delay=2.5, restart_after_pages=5, *args,
                 **kwargs):
        super(WgaspiderSpider, self).__init__(*args, **kwargs)

        # set parameters
        self.category = category
        self.typeof = typeof
        self.folder = folder
        self.max_site_load_delay = max_site_load_delay
        self.restart_after_pages = int(restart_after_pages)

        # init folder
        try:
            os.makedirs(self.folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # init web driver
        options = Options()
        options.headless = True
        profile = webdriver.FirefoxProfile()
        # profile.set_preference("browser.cache.disk.enable", True)
        # profile.set_preference("browser.cache.memory.enable", True)
        # profile.set_preference("browser.cache.offline.enable", True)
        # profile.set_preference("network.http.use-cache", True)
        self.driver = webdriver.Firefox(profile, options=options)
        self.driver.minimize_window()

    def restart_browser(self, response):
        # close driver
        self.driver.close()
        self.driver.quit()

        # init new driver
        options = Options()
        options.headless = True
        profile = webdriver.FirefoxProfile()
        # profile.set_preference("browser.cache.disk.enable", True)
        # profile.set_preference("browser.cache.memory.enable", True)
        # profile.set_preference("browser.cache.offline.enable", True)
        # profile.set_preference("network.http.use-cache", True)
        self.driver = webdriver.Firefox(profile, options=options)
        self.driver.minimize_window()

        # open website
        self.driver.get(response.url)

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
        # save actual page
        actual_page_number = 0

        search_done = self.get_through_search()

        while search_done:
            # get all image boxes from actual page
            image_boxes = self.driver.find_elements_by_xpath('/html/body/center[3]/table/tbody/tr')

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
                WebDriverWait(self.driver, self.max_site_load_delay).until(
                    ec.visibility_of_element_located((By.XPATH, '/html/frameset/frame[2]')))

                # switch to new frame in popup window
                frame = self.driver.find_element_by_xpath('/html/frameset/frame[2]')
                self.driver.switch_to.frame(frame)

                # wait
                WebDriverWait(self.driver, self.max_site_load_delay).until(
                    ec.visibility_of_element_located((By.XPATH, '/html/body/center/img')))

                # get and save the image
                image = self.driver.find_element_by_xpath('/html/body/center/img')
                src = image.get_attribute('src')
                urllib.request.urlretrieve(src, self.folder + str(actual_img_id) + '.jpg')

                # increment image id
                actual_img_id += 1

                # close popup window
                self.driver.close()

                # switch to main window
                self.driver.switch_to.window(window_before)

                # increment counter for images on the current page
                i += 1

            # increment page counter
            actual_page_number += 1

            # restart browser every few pages to hold ram footprint small
            if actual_page_number % self.restart_after_pages == 0:
                self.restart_browser(response)
                search_done = self.get_through_search()
                self.iterate_pages(actual_page_number)

        # close main window to end scraping
        self.driver.close()
        self.driver.quit()

    def iterate_pages(self, number_pages):
        for i in range(number_pages):
            # get next page button
            next_page_button = self.driver.find_elements_by_xpath(
                '/html/body/center[4]/table/tbody/tr/td/p/a')[-1]

            # switch to next page
            next_page_button.click()

            # wait for page to load
            self.driver.implicitly_wait(self.max_site_load_delay)
            # WebDriverWait(self.driver, self.max_site_load_delay).until(
            #    ec.visibility_of_element_located((By.XPATH, '/html/frameset/frame[2]')))

    def get_through_search(self):
        # hit enter button, which is 'hidden' from bots, by manuel click
        img_over_enter_button = self.driver.find_element_by_xpath('/html/body/table/tbody/tr[5]/td[1]/img')
        ActionChains(self.driver).move_to_element_with_offset(img_over_enter_button, 35, 33).click().perform()

        # Wait for javscript to load in Selenium
        WebDriverWait(self.driver, self.max_site_load_delay).until(
            ec.visibility_of_element_located((By.XPATH, '/html/frameset/frame[2]')))

        # switch to new frame
        frame = self.driver.find_element_by_xpath('/html/frameset/frame[2]')
        self.driver.switch_to.frame(frame)

        # click link to get to search engine
        WebDriverWait(self.driver, self.max_site_load_delay).until(
            ec.visibility_of_element_located((By.LINK_TEXT, 'Search Engine')))
        self.driver.find_element_by_link_text('Search Engine').click()

        # Wait for javscript to load in Selenium
        WebDriverWait(self.driver, self.max_site_load_delay).until(
            ec.visibility_of_element_located((By.XPATH, '/html/body/form/center/p[4]/font/input[1]')))

        # fill search engine forms
        select = Select(self.driver.find_element_by_name('form'))
        select.select_by_visible_text(self.category)
        select = Select(self.driver.find_element_by_name('type'))
        select.select_by_visible_text(self.typeof)

        # get search button and click it
        search_engine_button = self.driver.find_element_by_xpath(
            '/html/body/form/center/p[4]/font/input[1]')
        search_engine_button.click()

        # Wait for javscript to load in Selenium
        WebDriverWait(self.driver, self.max_site_load_delay).until(
            ec.visibility_of_element_located((By.XPATH, '/html/body/center[3]/table/tbody/tr')))

        return True
