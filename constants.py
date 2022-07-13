# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:21:50 2022

@author: Luciano
"""

seg_map = {
    r'[1-2][1-2]': 0, #hibernating
    r'[1-2][3-4]': 1, #at_Risk
    r'[1-2]5': 2, #cant_loose
    r'3[1-2]': 3, # about_to_sleep
    r'33': 4, # need_attention
    r'[3-4][4-5]': 5, #loyal_customers
    r'41': 6, #promising
    r'51': 7, #new_customers
    r'[4-5][2-3]': 8, #potential_loyalists
    r'5[4-5]': 9# champion
}

