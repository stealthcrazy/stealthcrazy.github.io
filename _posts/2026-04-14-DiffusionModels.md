---
title: "Building a Diffusion Model from Scratch"
tags: [diffusion-models, deep-learning]
---

### Introduction
In this Post , I will be doing documenting my journey researching Diffusion Models for Image Generation. Particularly, I will go into understanding the Mathematics behind the Model , Building the architecture in PyTorch and discussing how to undertake training the models on datasets like CIFAR10 and CELEB-A 128x128.
###  Looking at Maths  Behind Diffusion Models

To Begin, Diffusion Models are categorised as Latent variable Models of the form below

$$
\begin{aligned}
	p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T}) \; dx_{1:T} \qquad (1a)
\end{aligned}
$$

Here $x_{1} \dots x_{T}$ are known as the Latents which have the same dimensions as the data/ image we begin with ($x_{0}$).Moving on, $p_{\theta}(x_{0:T})$ is a joint distribution of all the latents and is referred to as the reverse process.  Specifically Equation 1 states  that to get back original $p_{\theta}(x_{0})$ we should integrate over all latents to eliminate the from the joint distribution $p_{\theta}(x_{0:T})$ which is intractable . Additionally the reverse process is a Markov Chain with learnt transitions( Gaussian based ) starting at $p(x_{T})$ so can be defined as the following

$$
\begin{aligned}
p_{\theta}(x_{0:T}) = p_{\theta}(x_{T})\prod_{t = 1} ^T p_{\theta}(x_{t-1}\mid x_{t}) \quad \quad (1b)  \\ 
p_{\theta}(x_{t-1}\mid x_{t}) =\mathcal{ N}(x_{t-1} ;\; \mu_{\theta}( x_{t},t) \:,\: \Sigma_{\theta}( x_{t},t) \: ) \quad (1c)
\end{aligned}
$$

The key differentiator for a diffusion model to that of a latent is the forward process ( Diffusion ) which is a fixed Markov chain that repeatedly adds noise to the data/image. The forward process is analogous to brownian motion in the idea that the image gets noisier with each time step in the forward diffusion process due to the variance scheduling of $\beta_{1} \dots \beta_{T}$ .  Below in Equation 2a is the definition of the forward process .

$$
\begin{aligned}
q(x_{1:T}\mid x_{0}) = \prod_{t = 1} ^T q(x_{t}\mid x_{t-1}) \quad \quad (2a)  \\ 
q(x_{t}\mid x_{t-1}) =\mathcal{ N}(x_{t} ;\; \sqrt{1-\beta_{t}}\: x_{t-1} , \;\beta_{t}I ) \quad (2b)
\end{aligned}
$$

Equation 2b defines how the Markov chain can be calculated using fixed gaussian noise to attain noisier versions of the data $x_{0}$ for a given time stamp $t$. 

Moving on to Training, since marginalisation of the joint probability distribution (Equation 1a) is not possible we look at the negative log likelihood instead. Below is derivation i wrote in LaTex following the steps for Evidence Bound Lower Bound (ELBO) .

#### ELBO

To begin

$$
\begin{aligned}  \\
 \log(\: p_{\theta}(x_{0})\:) &= \mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]  \qquad (3a)
\end{aligned}
$$

Using Bayes Rule we can write 

$$
 p_{\theta}(x_{0}) =  \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T}\mid x_{0})}
$$

and after substituting Back into (3) and then multiplying by $\frac{q(x_{1:T}\mid x_{0})}{q(x_{1:T}\mid x_{0})}$ we get

$$

\begin{aligned}  \\
 &= \mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T}\mid x_{0})} \cdot \frac{q(x_{1:T}\mid x_{0})}{q(x_{1:T}\mid x_{0})} )\:]  \\ \\
 &=\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) \cdot q(x_{1:T}\mid x_{0})}{q(x_{1:T}\mid x_{0}) \cdot p_{\theta}(x_{1:T}\mid x_{0})} )\:] 
\end{aligned} \qquad (3b) 
$$

To Further simplify we can split Expectation into two terms

$$
\begin{aligned}  \\
 &=\underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})})\:]}_{ELBO} +\underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{q(x_{1:T}\mid x_{0}) }{p_{\theta}(x_{1:T}\mid x_{0})})\:] }_{D_{KL}(q(x_{1:T}\mid x_{0})\: \mid \mid  \: p_{\theta}(x_{1:T}\mid x_{0}))} \\ \\ 
 &=\underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})})\:]}_{ELBO} + D_{KL}(q(x_{1:T}\mid x_{0})\: \mid \mid  \: p_{\theta}(x_{1:T}\mid x_{0}))
\end{aligned} \qquad (3c)
$$

Here the Second term is The KL divergence and it has a property that : 

$$
\begin{aligned} \\
D_{KL}(q(x_{1:T}\mid x_{0})\: \mid \mid  \: p_{\theta}(x_{1:T}\mid x_{0})) \geq 0
\end{aligned} 
$$

This fact then can be used to relate $\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ with ELBO and KL divergence as follows by rearranging

$$
	\begin{aligned} \\ \\
	\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]   - \:ELBO &=  \: D_{KL}(q(x_{1:T}\mid x_{0})\: \mid \mid  \: p_{\theta}(x_{1:T}\mid x_{0})) \:  \\ 
	\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]   - \:ELBO &\geq 0 \\
	\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]   &\geq ELBO \\
	\end{aligned} \qquad (3d)
$$

The gap between $\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ and ELBO is defined as the tightness of the bound. Additionally since the KL Divergence determines the distance between the prior (Q) and posterior (P) distributions the tighter the bound the better it approximates between the two distributions. This can be done by Maximising the ELBO. However we can do a trick to instead minimise the bound by multiplying by a negative sign. This makes it easier in PyTorch to Train 

$$
\begin{aligned} \\
\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]   &\geq \underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})})\:]}_{ELBO} \\ \\
\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: -\log(\:p_{\theta}(x_{0:T})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: -\log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})})\:]}_{ELBO} \\ \\
\end{aligned} \qquad (3e)
$$

This is the same as the Equation 3 in the Denoising Diffusion Probabilistic Model Paper. We can then use Equation 1b and 2a to write the parts of the ELBO as follows

$$
\begin{aligned} \\
\mathbb{E}_{q(x_{1:T}\mid x_{0})}[\: -\log(\:p_{\theta}(x_{0:T})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T}\mid x_{0})}\left[ \: -\log\left( \: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})} \right)\: \right]}_{ELBO} \\  \\

&= \mathbb{E}_{q(x_{1:T}\mid x_{0})}\left[ -\log\left(  p_{\theta}(x_{T})\prod_{t = 1} ^T \frac{p_{\theta}(x_{t-1}\mid x_{t})}{ q(x_{t}\mid x_{t-1})} \right) \right] \\ \\ \\
&= \mathbb{E}_{q(x_{1:T}\mid x_{0})}\left[ -\log(  p_{\theta}(x_{T}) - \sum_{t\geq 1} \log(\frac{p_{\theta}(x_{0:T}) }{q(x_{1:T}\mid x_{0})}) \right]
\end{aligned} \qquad (3f)
$$
