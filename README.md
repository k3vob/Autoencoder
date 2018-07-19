# Autoencoder

Autoencoders were first proposed by Geoffrey Hinton as a better, non-linear alternative to PCA for dimenstionality reduction of data sets. (Paper: https://www.cs.toronto.edu/~hinton/science.pdf)

This repo contains a deep autoencoder model built in TensorFlow with support for noisy and/or sparse input. The weights of the decoder can be set to be the direct transposes of the encoder weights using the `tiedWights` boolean parameter. This gives symmetric, tied weights between the encoder and decoder. Otherwise the encoder and decoder weights will be trained independently from each other.

Below shows the process of the model learning to take noisy/corrupted imaged data of handwritten letters as input, encode the data to ~6% of its original size, and then reconstruct it while removing all noise.

## To Do:

- ~Add functionality to tie decoder weights to encoder weights.~
- ~Add denoising abilities.~
- Add functionality to create a stacked autoencoder by training the model layer-by-layer.
- ~Add support for sparse input data.~
- Add regularisation techniques.

### Original, encoded, and reconstructed/denoised images:

![alt text](https://i.imgur.com/QyES7ct.png "autoencoder")
