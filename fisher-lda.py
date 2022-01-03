from matplotlib import pyplot as plt
import numpy as np
from dataset import Dataset


class FisherLDA():
    def __init__(self):
        self.dataset = Dataset()
        self.number_of_component = None
        self.mean_feature = None
        self.centered_sample = None
        self.eigenvectors = None
        self.transformed_sample = None
        self.reconstructed_sample = None
        self.number_of_visual_sample = None
        self.visualize_sample = None