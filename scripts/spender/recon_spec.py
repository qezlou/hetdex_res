import numpy as np
from spender import SpectrumAutoencoder, SpeculatorActivation
import torch
from accelerate import Accelerator
from spender.data.hetdex import HETDEX
from spender import SpectrumAutoencoder, SpeculatorActivation
import h5py
import argparse
import os.path as op


class Reconstructor:
    """
    For a trained model this module froward path the model on the gien spectra files
    and saves the reconstructed spectra and latent representations to an output file.
    """
    def __init__(self, data_dir, spec_file, model_file, wave_obs=None, which='both'):
        self.data_dir = data_dir
        self.spec_file = spec_file
        self.wave_obs = wave_obs
        self.model_path = op.join(data_dir, 'models', model_file)
        self.instrument, self.dataloader = self.load_train_val_data()
        print(f'Loaded data with {len(self.dataloader.dataset)} spectra.')
        self.model, self.losses = self.load_model(self.model_path)
        self.model, self.dataloader = Accelerator().prepare(self.model, self.dataloader)
    
    def load_train_val_data(self, wave_obs=None):
        instrument = HETDEX(wave_obs=wave_obs)
        dataloader = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="train", batch_size=16384, seed=42, split_ratio=1.0)
        return instrument, dataloader

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

    def load_model(self, model_path):
        device = self.instrument.wave_obs.device
        print(f"Loading model from {model_path} onto device {device}")
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device)
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
        print(f'GPU available: {torch.cuda.is_available()}', flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, ckpt['losses']


    def reconstruct_spectra(self):
        """Reconstruct spectra using the loaded model.

        Parameters
        ----------
        spectra: torch.Tensor
            Input spectra to reconstruct, shape (n_samples, n_wavelengths)

        Returns
        -------
        recon_spectra: torch.Tensor
            Reconstructed spectra, shape (n_samples, n_wavelengths)
        """
        with torch.no_grad():
            all_latents = []
            all_recon = []
            all_inds = []
            self.model.eval()
            for b, batch in enumerate(self.dataloader):
                print(f'Reconstructing batch {b+1}/{len(self.dataloader)}', flush=True)
                spec, w,_, ind = batch
                print(f'Batch spec shape: {spec.shape}', flush=True)
                latents, recon_spectra, _, _ = self.model._forward(spec)
                if b == 0:
                    all_latents = latents.cpu()
                    all_recon = recon_spectra.cpu()
                    all_inds = ind.cpu()
                else:
                    all_latents = torch.cat((all_latents, latents.cpu()), dim=0)
                    all_recon = torch.cat((all_recon, recon_spectra.cpu()), dim=0)
                    all_inds = torch.cat((all_inds, ind.cpu()), dim=0)
            all_latents = all_latents.numpy()
            all_recon = all_recon.numpy()
            all_inds = all_inds.numpy()
        return all_latents, all_recon, all_inds
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data file directory")
    parser.add_argument("spec_file", help='fiber spectra file to use')
    parser.add_argument("model_file", help="path to trained model file")
    parser.add_argument("output_file", help="output file name for reconstructed spectra")
    args = parser.parse_args()
    reconstructor = Reconstructor(
        data_dir=args.data_dir,
        spec_file=args.spec_file,
        model_file=args.model_file,
        wave_obs=None,
        which='both'
    )
    latents, recon, inds = reconstructor.reconstruct_spectra()
    out_path = op.join(args.data_dir, 'recon', args.output_file)
    with h5py.File(out_path, 'w') as f:
            f.create_dataset('latents', data=latents)
            f.create_dataset('recon_spectra', data=recon)
            f.create_dataset('inds', data=inds)