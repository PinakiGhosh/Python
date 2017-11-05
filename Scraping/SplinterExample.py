#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 23:34:23 2017

@author: pinaki
"""

from splinter import Browser

browser = Browser()
browser.visit('http://google.com')
browser.fill('q', 'splinter - python acceptance testing for web applications')
browser.find_by_name('btnG').click()

if browser.is_text_present('splinter.readthedocs.io'):
    print ("Yes, the official website was found!")
else:
    print ("No, it wasn't found... We need to improve our SEO techniques")

browser.html
browser.quit()