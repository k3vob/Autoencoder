# Autoencoder

Deep autoencoder model built in TensorFlow with support for noisy and/or sparse input. The weights of the decoder can be set to be the direct transposes of the encoder weights using the `tiedWights` boolean parameter. This gives symmetric, tied weights between the encoder and decoder. Otherwise the encoder and decoder weights will be trained independently from each other.

## To Do:

- ~Add functionality to tie decoder weights to encoder weights.~
- ~Add denoising abilities.~
- Add functionality to create a stacked autoencoder by training the model layer-by-layer.
- ~Add support for sparse input data.~
- Add regularisation techniques.

### Original, encoded, and reconstructed/denoised images:

![alt text](https://i.imgur.com/QyES7ct.png "autoencoder")
