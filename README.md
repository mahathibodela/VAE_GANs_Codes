# VAE_GANs_Codes

## VAE

**ASSUMPTION* *: Data is formed based on the underlying distribution of the latent variables, so creating that space leads to generating images

**PROBLEM** : p(z/x) is often intractable, due to the integration in the denominator so cant be caluclated directly. 

**PROPOSAL** : Approximating p(z/x) with another distribution q(z) through neural networks(encoder), then generating image from the space with the use of repameterisation technique

**OBJECTIVE** : KL divergenec btw q(z) & p(z/x) should be as minimum as possiable, Reconstruction Loss for output of decoder to be same similar as input image


## GAN

**PROBLEM** : Though, VAE gave a solution approximation. Approximate Generative models are not being that effective.

**PROPOSAL** : Rather than trying to model probabilistic computations bcz of which it is not being effective, use an adversarial neural network where generator maps data distribution to model distribution and discriminator classifies wheather the input came from model distribution or data distribution

**OBJECTIVE** : Both gen & disc wants to win aganist each other, so disc goal is to classify crtly gen goal is to generate images indistinguishable form the real images. As, it is a game play equilibrium attains when disc output 0.5 propbability for every input image.

##
""
