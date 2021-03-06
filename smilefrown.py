# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:10:37 2020

@author: abbme
"""

from generate_synthetic import load_img 
from generate_synthetic import gs_to_data
from generate_synthetic import generate_arima_ts
from generate_synthetic import reduce_coldim
import numpy as np

def l_smile_frown(col_red):
    s1 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\s1.jpg")
    s2 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\s2.jpg")
    s3 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\s3.jpg")
    s4 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\s4.jpg")
    f1 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\f1.jpg")
    f2 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\f2.jpg")
    f3 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\f3.jpg")
    f4 = load_img(path=r"C:\Abbas\Projects\Thesis\DataPics\FrownSmile\f4.jpg")
    imglist = [s1,s2,s3,s4,f1,f2,f3,f4]
    data = []
    for img in imglist:
        data.append(generate_arima_ts(gs_to_data(reduce_coldim(img,col_red)),n=25))
        
    data = np.vstack(data)
    m,_ = data.shape
    return(np.append(data, np.concatenate((np.zeros((int(m/2),1)),np.ones((int(m/2),1)))), axis=1))