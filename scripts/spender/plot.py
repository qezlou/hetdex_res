"""
Plot spender plots
"""
import numpy as np
from spender import SpectrumAutoencoder, SpeculatorActivation
import torch
from matplotlib import pyplot as plt
import corner
from spender.data.hetdex import HETDEX

class HetSpenderPlot():

    def __init__(self, data_dir, wave_obs=None):
        self.data_dir = data_dir
        self.wave_obs = wave_obs
        self.instrument, self.trainloader, self.validloader = self.load_train_val_data()

    def load_train_val_data(self, data_dir, wave_obs=None):

        instrument = HETDEX(wave_obs=wave_obs)

        trainloader = instrument.get_data_loader(data_dir, which="train", batch_size=1024)
        validloader = instrument.get_data_loader(data_dir, which="valid", batch_size=1024)
        return instrument, trainloader, validloader


    def plot_diagnostic(saved_file, wave_rest= torch.arange(3470, 5542, 2, dtype=torch.float32), latents=2):

        

