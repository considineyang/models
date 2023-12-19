# Deep contextualized word representations paper: https://arxiv.org/pdf/1802.05365.pdf
# Tutorial: https://github.com/MorvanZhou/NLP-Tutorials/blob/master/ELMo.py
from tensorflow import keras
import tensorflow as tf 
import time 
import os

class ELMo(keras.Model):
    def __init__(self, v_dim, emb_dim, units, n_layers, lr):
        super().__init__()
        self.n_layers = n_layers
        self.units = units
        
