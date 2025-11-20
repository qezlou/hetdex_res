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
    def __init__(self, data_dir, spec_file, model_file, wave_obs=None, which='both',frac_each_file=0.2):
        self.data_dir = data_dir
        self.spec_file = spec_file
        self.wave_obs = wave_obs
        self.model_path = op.join(data_dir, 'models', model_file)
        self.instrument, self.trainloader, self.validloader = self.load_train_val_data(which=which, frac_each_file=frac_each_file)
        self.model, self.losses = self.load_model(self.model_path)
        self.model, self.trainloader, self.validloader = Accelerator().prepare(self.model, self.trainloader, self.validloader)
    
    def load_train_val_data(self, wave_obs=None, which='both', frac_each_file=0.2):

        instrument = HETDEX(wave_obs=wave_obs)
        if which is not 'both':
            trainloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="train", batch_size=16384, frac_each_file=frac_each_file)
            validloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="valid", batch_size=16384, frac_each_file=frac_each_file)
            return instrument, trainloader, validloader
        else:
            trainloader, _ = instrument.get_data_loader(dir=self.data_dir, file_name=self.spec_file, which="both", batch_size=16384, frac_each_file=frac_each_file)
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

    def to_cpu_list(self,x):
            # x can be a tensor or a list/tuple of tensors
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return [t.detach().cpu() if isinstance(t, torch.Tensor) else t for t in x]

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

        train_latents, train_recon_spectra, train_og_spectra = [], [], []
        valid_latents, valid_recon_spectra, valid_og_spectra = [], [], []

        with torch.no_grad():
            self.model.eval()
            for b, batch in enumerate(self.trainloader):
                spec, w, _ = batch
                latent, recon_spectra, _, _ = self.model._forward(spec)
                train_latents.extend(self.to_cpu_list(latent))
                train_recon_spectra.extend(self.to_cpu_list(recon_spectra))
                train_og_spectra.extend(self.to_cpu_list(spec))
            if self.validloader is None:
                return np.array(train_latents), np.array(train_recon_spectra), np.array(train_og_spectra), None, None, None
            
            else:
                for b, batch in enumerate(self.validloader):
                    spec, w, _ = batch
                    latent, recon_spectra, _, _ = self.model._forward(spec)
                    valid_latents.extend(self.to_cpu_list(latent))
                    valid_recon_spectra.extend(self.to_cpu_list(recon_spectra))
                    valid_og_spectra.extend(self.to_cpu_list(spec))
                return np.array(train_latents), np.array(train_recon_spectra), np.array(train_og_spectra), np.array(valid_latents), np.array(valid_recon_spectra), np.array(valid_og_spectra)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data file directory")
    parser.add_argument("spec_file", help='fiber spectra file to use')
    parser.add_argument("model_file", help="path to trained model file")
    parser.add_argument("output_file", help="output file name for reconstructed spectra")
    parser.add_argument("frac_to_use", help="fraction of data to use from each file", type=float, default=0.1)
    args = parser.parse_args()
    reconstructor = Reconstructor(
        data_dir=args.data_dir,
        spec_file=args.spec_file,
        model_file=args.model_file,
        wave_obs=None,
        which='both',
        frac_each_file=args.frac_to_use
    )
    latents_train, recon_spectra_train, og_spectra_train, latents_valid, recon_spectra_valid, og_spectra_valid = reconstructor.reconstruct_spectra()
    out_path = op.join(args.data_dir, 'recon', args.output_file)
    with h5py.File(out_path, 'w') as f:
            f.create_dataset('train_latents', data=latents_train)
            f.create_dataset('train_recon_spectra', data=recon_spectra_train)
            f.create_dataset('train_og_spectra', data=og_spectra_train)
            if latents_valid is not None:
                f.create_dataset('valid_latents', data=latents_valid)
                f.create_dataset('valid_recon_spectra', data=recon_spectra_valid)
                f.create_dataset('valid_og_spectra', data=og_spectra_valid)