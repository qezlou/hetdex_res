"""
Plot spender plots
"""
import numpy as np
from spender import SpectrumAutoencoder, SpeculatorActivation
import torch
from matplotlib import pyplot as plt
import corner
from spender.data.hetdex import HETDEX
from spender import SpectrumAutoencoder, SpeculatorActivation

class HetSpenderPlot():

    def __init__(self, data_dir, spec_file, model_path, wave_obs=None, which='both'):
        self.data_dir = data_dir
        self.spec_file = spec_file
        self.wave_obs = wave_obs
        self.model_path = model_path
        self.instrument, self.trainloader, self.validloader = self.load_train_val_data(which=which)
        self.model, self.losses = self.load_model(model_path)


    def load_train_val_data(self, wave_obs=None, which='both'):

        instrument = HETDEX(wave_obs=wave_obs)
        if which is not 'both':
            trainloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="train", batch_size=128)
            validloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="valid", batch_size=128)
            return instrument, trainloader, validloader
        else:
            trainloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="both", batch_size=128)
            return instrument, trainloader, None




    def infer_decoder_hyperparams(self, sd):
        """
        Infer n_latent and decoder n_hidden from the decoder MLP weights
        in a SpectrumAutoencoder checkpoint state dict.
        """
        # pick only 2D weight tensors of the decoder MLP (Linear layers)
        dec_w_keys = [
            k for k, v in sd.items()
            if k.startswith("decoder.mlp")
            and k.endswith("weight")
            and v.ndim == 2
        ]
        # Sort by the layer index in `decoder.mlp.X.weight`
        dec_w_keys = sorted(dec_w_keys, key=lambda k: int(k.split('.')[2]))

        if not dec_w_keys:
            raise RuntimeError("No decoder MLP linear weights found in state_dict.")

        dec_shapes = [sd[k].shape for k in dec_w_keys]
        # MLP: n_ = [n_in, *n_hidden, n_out]
        # weight[i] shape = (n_[i+1], n_[i])

        # First layer: (n_hidden[0], n_latent)
        first_out, first_in = dec_shapes[0]
        n_latent = first_in

        # Hidden sizes: all but last layer's output
        # Last linear layer maps to n_out = len(wave_rest)
        decoder_n_hidden = [out for (out, inp) in dec_shapes[:-1]]

        # Output size (len(wave_rest)), not strictly needed except as a sanity check
        output_dim = dec_shapes[-1][0]

        return n_latent, tuple(decoder_n_hidden), output_dim

    def load_model(self, model_path, map_location="cpu"):
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=map_location)
        # Some checkpoints are {'model': sd, 'losses': ...}, others are just sd
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt

        # Infer decoder hyperparameters
        n_latent, decoder_n_hidden, output_dim = self.infer_decoder_hyperparams(sd)

        # Recover wave_rest from the saved buffer
        if "decoder.wave_rest" not in sd:
            raise RuntimeError("decoder.wave_rest not found in state_dict; cannot recover wave_rest.")
        wave_rest = sd["decoder.wave_rest"]  # 1D tensor

        # Optional: sanity check encoder
        # encoder_hidden = check_encoder(sd, n_latent)
        # print("Encoder hidden layers:", encoder_hidden)

        act = (SpeculatorActivation(decoder_n_hidden[0]), 
           SpeculatorActivation(decoder_n_hidden[1]), 
           SpeculatorActivation(decoder_n_hidden[2]), 
           SpeculatorActivation(len(self.instrument.wave_obs), plus_one=False))
        
        # Construct model with correct hyperparameters
        model = SpectrumAutoencoder(
            instrument=self.instrument,
            wave_rest=wave_rest,
            n_latent=n_latent,
            n_hidden=decoder_n_hidden,
            act=act,  # use default SpeculatorActivation; params will be overwritten
            )

        # Load weights
        model.load_state_dict(sd)
        model.eval()
        return model, np.array(ckpt['losses'])


    def plot_loss(self):

        pass