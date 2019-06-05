# DRAW-GAN
DRAW-GAN:   An Adversarial Procedure For Sequential Image Generation

Varitional autoencoders (VAE), as generative models, generate image by sampling from approximated Guassian distribution. The generated image seems to have noise due to limited number of features captured by VAE during stochastic graident decent learning. DRAW generates image, based on VAE and RNN using attention mechanism. In this article, the concept of adversarial procedure is incorporated into the DRAW network to overcome the problem of noise in image and for better approximation of prior. By replacing the evidence lower bound (ELBO) loss function in DRAW with the adversarial loss function and adding discriminator network into DRAW, DRAW-GAN is developed. The performance of DRAW-GAN for generating images is compared with three differnt methods using two data sets, namely OCR Telugu and MNIST. The proposed DRAW-GAN performs better than the remaining methods.

The Project contains the following files :- -Config.py --- congigurations of neural network. -Train.py --- Training Procedure of network. -Draw_model.py --- Defining Model of Network. -Utility.py --- Utilities used in model. -Generated_Images --- This folder contains the output of this model. -edited Draw.pdf --- The paper explaining Model

This project is done by me and i am the first author. The Paper is under review so please it share any where.
