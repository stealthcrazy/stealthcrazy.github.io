---
title: "Building a Diffusion Model from Scratch"
tags: [diffusion-models, deep-learning]
---

### Introduction
In this Post , I will be  documenting my journey researching Diffusion Models for Image Generation.As of writing this first post I am a first year UG studying CS and and first time working with LaTex.   Particularly, I will go into understanding the Mathematics behind the Model , Building the architecture in PyTorch and discussing how to undertake training the models on datasets like CIFAR10 and CELEB-A 128x128. Additionally, I will try to not take any shortcuts and showcase how equations were derived from my perspective.However there is a lot to cover and will be prone to mistakes.
###  Looking at Maths  Behind Diffusion Models

To Begin, Diffusion Models are categorised as Latent xvariable Models of the form below
$$
\begin{align}
	p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T}) \; dx_{1:T} \qquad (1a)
\end{align}
$$
Here $x_{1} \dots x_{T}$ are known as the Latents which have the same dimensions as the data/ image we begin with ($x_{0}$).Moving on, $p_{\theta}(x_{0:T})$ is a joint distribution of all the latents and is referred to as the reverse process.  Specifically Equation 1 states  that to get back original $p_{\theta}(x_{0})$ we should integrate over all latents to eliminate the from the joint distribution $p_{\theta}(x_{0:T})$ which is intractable . Additionally the reverse process is a Markov Chain with learnt transitions( Gaussian based ) starting at $p(x_{T})$ so can be defined as the following
$$
\begin{align}
p_{\theta}(x_{0:T}) = p_{\theta}(x_{T})\prod_{t = 1} ^T p_{\theta}(x_{t-1} \mid x_{t}) \quad \quad (1b)  \\ 
p_{\theta}(x_{t-1} \mid x_{t}) =\mathcal{ N}(x_{t-1} ;\; \mu_{\theta}( x_{t},t) \:,\: \Sigma_{\theta}( x_{t},t) \: ) \quad (1c)
\end{align}
$$
The key differentiator for a diffusion model to that of a latent is the forward process ( Diffusion ) which is a fixed Markov chain that repeatedly adds noise to the data/image. The forward process is analogous to brownian motion in the idea that the image gets noisier with each time step in the forward diffusion process due to the variance scheduling of $\beta_{1} \dots \beta_{T}$ .  Below in Equation 2a is the definition of the forward process .

$$
\begin{align}
q(x_{1:T} \mid x_{0}) = \prod_{t = 1} ^T q(x_{t} \mid x_{t-1}) \quad \quad (2a)  \\ 
q(x_{t} \mid x_{t-1}) =\mathcal{ N}(x_{t} ;\; \sqrt{1-\beta_{t}}\: x_{t-1} , \;\beta_{t}I ) \quad (2b)
\end{align}
$$

Equation 2b defines how the Markov chain can be calculated using fixed gaussian noise to attain noisier versions of the data $x_{0}$ for a given time stamp $t$. 

Moving on to Training, since marginalisation of the joint probability distribution (Equation 1a) is not possible we look at the negative log likelihood instead. Below is derivation i wrote in LaTex following the steps for Evidence Lower Bound (ELBO) .

#### ELBO

To begin there are a few things to note which puzzled me when I began. 

$$
\begin{aligned}
	1)&\quad \mathbb{E}_{q(x)}\left[p(x) \right] = E_{x \sim q(x)}\left[p(x) \right] = \int p(x)\cdot q(x)dx \\
	2)& \quad  p_{\theta}(x_{0:T}) =  p_{\theta}(x_{0},\dots ,x_{T})\\
	3)& \quad D_{KL}(P  \mid  \mid  \: Q) = \sum_{x \in X} P(x)\log\left( \frac{P(x)}{Q(x)} \right)

	
	
\end{aligned}

$$
- 1) states that it is the expectation taken with respect to destitution q(x). This is needed to take expectations when more than one variable involved.
- 2) states the notation used by the DDPM paper to represent joint probability distributions.
- 3) states the formula for Kullback-Leibler Divergence which measures the distances between two probability distribution.

Having that sorted we can now move on to the following start:
$$
\begin{align}  \\ 
 \log(\: p_{\theta}(x_{0})\:)  & =  log(\: p_{\theta}(x_{0})) \cdot \int q(x_{1:T}) \: dx_{1:T} \\
 &= \int log(\: p_{\theta}(x_{0})) \cdot q(x_{1:T}) \: dx_{1:T}  \\  
 &= \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]  \qquad \qquad \\

\end{align} \qquad(3a)
$$

Using Conditional Probability we can write 
$$
 p_{\theta}(x_{0}) =  \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T} \mid x_{0})}
$$
and after substituting Back into (3) and then multiplying by $\frac{q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0})}$ we get
$$
\begin{align}  \\
 &= \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T} \mid x_{0})} \cdot \frac{q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0})} )\:]  \\ \\
 &=\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) \cdot q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0}) \cdot p_{\theta}(x_{1:T} \mid x_{0})} )\:] 
\end{align} \qquad (3b) 

$$
To Further simplify we can split Expectation into two terms

$$
\begin{align}  \\
 &=\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} +\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{q(x_{1:T} \mid x_{0}) }{p_{\theta}(x_{1:T} \mid x_{0})})\:] }_{D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0}))} \\ \\ 
 &=\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} + D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0}))
\end{align} \qquad (3c)
$$

Here the Second term is The KL divergence and it has a property that : 
$$
\begin{align} \\
D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0})) \geq 0
\end{align} 
$$
This fact then can be used to relate $\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ with ELBO and KL divergence as follows by rearranging
$$
\begin{align} \\ \\
 \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   - \:ELBO &=  \: D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0})) \:  \\ 
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   - \:ELBO &\geq 0 \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   &\geq ELBO \\
\end{align} \qquad (3d)
$$
The gap between $\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ and ELBO is defined as the tightness of the bound. Additionally since the KL Divergence determines the distance between the posterior (Q) and prior(P) distributions the tighter the bound the better it approximates between the two distributions. This can be done by Maximising the ELBO. However we can do a trick to instead minimise the bound by multiplying by a negative sign. This makes it easier in PyTorch to Train 
$$
\begin{align} \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   &\geq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} \\ \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\:p_{\theta}(x_{0})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{-ELBO} \\ \\
\end{align} \qquad (3e)
$$
This is the same as the Equation 3 in the Denoising Diffusion Probabilistic Model Paper. We can then use Equation 1b and 2a to write the parts of the ELBO as follows
$$
\begin{align} \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\:p_{\theta}(x_{0})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ \: -\log\left( \: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})} \right)\: \right]}_{ELBO} \\  \\

&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(  p_{\theta}(x_{T})\prod_{t = 1} ^T \frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right) \right] \\ \\ \\
&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p_{\theta}(x_{T})) - \sum_{t\geq 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right] = L
\end{align} \qquad (3f)$$
#### The Reparametrisation Trick
The authors of the Auto-encoding Variational Bayes , suggest a Trick to solve the problem when backpropogating through the network. Since the ELBO is an Expectation taken w.r.t the distribution $q(x_{1:T} \mid x_{0})$ , calculating  gradients  for the parameters associated  with prior distribution are  difficult to obtain but in the case of continuous latent variables they suggest reparametrisation. This is because of the stochastic nature of the Latents which prevents backpropgation and gradient calculation of the node in the graph. 

This can be done by introducing a new random variable $\epsilon \sim \mathcal{N}(0,I)$ and this can then be used  as follows for equation 2b: 
$$
\begin{aligned}
	x_{t} =   (\sqrt{1-\beta_{t}} \: \cdot x_{t-1})  \: + (\sqrt{  \beta_{t}} \cdot \:\epsilon) 
\end{aligned} \qquad (4a)
$$

Since $q(x_{1:T} \mid x_{0}) = \prod_{t = 1} ^T q(x_{t} \mid x_{t-1})$ we can now use the  (4a) to now write
$$
\begin{aligned}
	x_{t} &= \prod_{k=1} ^t \sqrt{1-\beta_{k} } \: \: \cdot x_{0} + \sum_{k=1}^t \:\left[ \prod_{l=k+1}^t\sqrt{1-\beta_{l}} \:  \right] \cdot \sqrt{\beta_{k} }\epsilon_{k} \\ 
& \text{using } \bar{\alpha_{t}} = \prod_{k=1} ^t \alpha_{k} \quad \text{and} \quad \alpha_{t} =  1-\beta_{t}\\ 
x_{t} &= \sqrt{  \bar{\alpha_{t}}} x_{0} + \sum_{k=1}^t \:\sqrt{ \frac{\bar{\alpha_{t}}}{\bar{\alpha_{k}}} } \cdot \sqrt{\beta_{k} }\epsilon_{k}\\
Mean &= \sqrt{ \bar{\alpha_{t}}}x_{0}  \qquad Var = \left[\sum_{k=1}^t \:\frac{\bar{\alpha_{t}}}{\bar{\alpha_{k}}}  \cdot \beta_{k} \right]
\end{aligned}
$$
The Mean is taken directly from the expanded relation for $x_{t}$ and the Variance for the sum of independent multivariate Gaussians (sampled from Normal : $\epsilon \sim \mathcal{N}(0,I)$) is the sum of individual variances and hence $Var = \sum_{k=1}^t \:\frac{\bar{\alpha_{t}}}{\bar{\alpha_{k}}}  \cdot \beta_{k}$ . This can be then further simplified :
$$
\begin{aligned}
	Var &= \sum_{k=1}^t \:\frac{\bar{\alpha_{t}}}{\bar{\alpha_{k}}}  \cdot \beta_{k} \\
	&= \bar{\alpha_{t}}\sum_{k=1}^t \:\frac{\beta_{k}}{\bar{\alpha_{k}}} \\
	&= \bar{\alpha_{t}}\sum_{k=1}^t \:\frac{1-\alpha_{k}}{\bar{\alpha_{k}}} \\ 
	
	& = \bar{\alpha_{t}} \left[ \sum_{k=1}^t \: \frac{1}{\bar{\alpha_{k}}} -\frac{1}{\bar{\alpha_{k-1}}} \right]  \qquad \qquad \text{Note:} \; \bar{\alpha_{0}}=1\\
	&= \bar{\alpha_{t}} \left[\frac{1}{\bar{\alpha_{t}}} -1 \right] \qquad \qquad \qquad \qquad  \\
	&= 1 - \bar{\alpha_{t}}

\end{aligned} (4b)
$$
The second to last step is solving a telescoping series which results in terms canceling out.So finally we can write 

$$
\begin{aligned}
	q(x_{t} \mid x_{0}) &= \mathcal{ N}(x_{t} ;\; \sqrt{\bar{\alpha_{t}}}\: x_{0} , \;(1-\bar{\alpha_{t}})I ) \qquad \qquad \\ \\
	x_{t} &=   \sqrt{\bar{\alpha_{t}}}\: x_{0}   \: + (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon) 
\end{aligned} (4c)
$$
### Reformulation for Efficient Training
The DDPM paper further optimises the Loss L to allow for better training. We First revisit 3f.
$$
\begin{align}
	L &=  \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ \: -\log\left( \: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})} \right)\: \right]}_{ELBO} \\  
	&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p_{\theta}(x_{T})) - \sum_{t\geq 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right]  \\  
	&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p_{\theta}(x_{T})) - \sum_{t > 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})}- \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right]  \qquad (5a)\\ \\ 

	&\text{using Baye's} \to \quad q(x_{t} \mid x_{t-1}) = q(x_{t}\mid x_{t-1},x_{0}) = \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t},x_{0})}{q(x_{t-1},x_{0})} \\ \\
	&= \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t}\mid x_{0})\cdot q(x_{0})}{q(x_{t-1}\mid x_{0})\cdot q(x_{0})} = \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t}\mid x_{0})}{q(x_{t-1}\mid x_{0})} \\ \\
    &\text{Substituting in (5a) we can obtain} \\ \\ 
	
	&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p_{\theta}(x_{T})) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}\cdot \frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) })\right] - \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right]  \; (5b)\\ \\  
	&\text{Focusing on the summation } \\ \\
	
&\sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}\cdot \frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) })\right] = \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}) \right] +\sum_{t > 1} \left[\log(\frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) }) \right] \\ \\
	
\end{align}
$$
The right hand side of the summation can further be then simplified as it is a telescoping series.
$$
\begin{aligned}
	\sum_{t > 1} \left[\log\left( \frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) } \right) \right] &= \sum_{t > 1} \left[\log(q(x_{t-1}\mid x_{0})) - \log({q(x_{t}\mid x_{0}) }) \right]\\ 
	&= \log(q(x_{1} \mid x_{0}))-\log(q(x_{T} \mid x_{0})) = \log(q(x_{1} \mid x_{0}))+\log\left(\frac{1}{q(x_{T} \mid x_{0})}\right)
\end{aligned}
$$
We can now use this result in (5b) 
$$
\begin{aligned}
	&\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p_{\theta}(x_{T})) - \log(q(x_{1} \mid x_{0}))- \log\left(\frac{1}{q(x_{T} \mid x_{0})}\right) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})})\right] - \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right] \\
&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(   \frac{p_{\theta}(x_{T})}{q(x_{T} \mid x_{0})} \right) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})})\right] - \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \; \right]\\
&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(   \frac{p_{\theta}(x_{T})}{q(x_{T} \mid x_{0})} \right) \right] + \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[\sum_{t > 1} \left[\log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})})\right]\right] - \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[- \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \;\right] \\ 
\end{aligned} 
$$
Focusing on Middle Expectation we can write
$$
\begin{aligned}
	\sum_{t > 1}\left[ \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[\log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}\right]\right]&=\sum_{t > 1} \left[\int\dots \int q(x_{1:T}  \mid x_{0}) \cdot \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \; dx_{1}\dots dx_{T}   \right]
\end{aligned}


$$
The integral can be simplified by integrating out parts that don't depend on t , t-1 
$$
\begin{aligned}
	=\sum_{t > 1} \left[\int\int q(x_{t-1},x_{t}  \mid x_{0}) \cdot \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \; dx_{t-1} \:dx_{t}   \right]
\end{aligned}
$$
