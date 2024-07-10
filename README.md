# VAE_GANs_Codes

This repo contains the implementation of various models which lays the fondation for the modern GenAi application in Vision field.

## VAE

<u>**ASSUMPTION**<u/>: Data is formed based on the underlying distribution of the latent variables, so creating that space leads to generating images.

<u>**PROBLEM**<u/> : p(z/x) is often intractable, due to the integration in the denominator so cant be caluclated directly. 

<u>**PROPOSAL**<u/> : Approximating p(z/x) with another distribution q(z) through neural networks(encoder), then generating image from the space with the use of repameterisation technique.

<u>**OBJECTIVE**<u/> : KL divergenec btw q(z) & p(z/x) should be as minimum as possiable, Reconstruction Loss for output of decoder to be same similar as input image.


## GAN

**PROBLEM** : Though, VAE gave approximation as solution. Approximate Generative models are not being that effective.

**PROPOSAL** : Rather than trying to model probabilistic computations bcz of which it is not being effective, use an adversarial neural network where generator maps data distribution to model distribution and discriminator classifies wheather the input came from model distribution or data distribution.

**OBJECTIVE** : Both gen & disc wants to win aganist each other, so disc goal is to give low propablity for fake images and high probability for real images, gen goal is to generate images such that disc gives it a high propobility. As, it is a game play equilibrium attains when disc output 0.5 propbability for every input image.

## DCGAN

**Refinement** : It is a refinement of GAN. Introduced CNN in building GAN.

## PIX2PIX

**PROBLEM** : There are various models for image to image translation bt, most of them are task specific. As the set up for image to image translation in every task, a more generalised approch would be appreciable.

**PROPOSAL** : GAN follows an adveserial process where gen aim is to generate real images and disc aim is to differentiate the input no matter what the task is, that means what ever the task there is no need of explicitly mentioning loss function. They exploited this detail of the GAN and proposed a new version GAN called Conditional GAN. In this gen outputs the translation of the source image to target image and the output of the disc is conditioned over source image.

**OBJECTIVE** : same as of GAN but to make sure that the output of the gen is as near as possiable to the target image, we use L1 loss, pass a pairs of real source & target images and real source & fake target images to the discriminator

## CYCLE GAN

**PROBLEM** : It is hard to get paired images for all image to image translation tasks. Though PIX2PIX works good bt its also needs paired dataset. An approch which can leverage single domain data would be appreciable

**PROPOSAL**: Just like pix2pix it also uses GANs, but with an extra loss that is Cycle loss. Adding this loss changed the whole play.

**OBJECTIVE**: same as of GAN but to make sure that the output of the gen is as near as possiable to the target domain we use cycle loss

