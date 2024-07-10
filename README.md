# VAE_GANs_Codes

## VAE

ASSUMPTION: Data is formed based on the underlying distribution of the latent variables, so creating that space leads to generating images

PROBLEM: p(z/x) is often intractable, due to the integration in the denominator so cant be caluclated directly. 

PROPOSAL: Approximating p(z/x) with another distribution q(z) through neural networks(encoder), then generating image from the space with the use of repameterisation technique

OBJECTIVE: KL divergenec btw q(z) & p(z/x) should be as minimum as possiable, Reconstruction Loss for output of decoder to be same similar as input image


## GAN

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014. GANs consist of two neural networks, the generator and the discriminator, which are trained simultaneously through adversarial processes. The generator creates synthetic data, while the discriminator attempts to differentiate between real and generated data.

