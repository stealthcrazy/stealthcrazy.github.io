---
layout: post
title: "Building a Diffusion Model from Scratch"
tags: [diffusion-models, deep-learning]
---

### Introduction
In this Post , I will be  documenting my journey researching Diffusion Models for Image Generation.As of writing this first post I am a first year UG studying CS and and first time working with LaTex.   Particularly, I will go into understanding the Mathematics behind the Model , Building the architecture in PyTorch and discussing how to undertake training the models on datasets like CIFAR10 and CELEB-A 128x128. Additionally, I will try to not take any shortcuts and showcase how equations were derived from my perspective.However there is a lot to cover and will be prone to mistakes.

###  Looking at Maths  Behind Diffusion Models

To Begin, Diffusion Models are categorised as Latent variable Models of the form below

$$

\begin{aligned}
	p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T}) \; dx_{1:T} \qquad (1a)
\end{aligned}

$$

Here $x_{1} \dots x_{T}$ are known as the Latents which have the same dimensions as the data/ image we begin with ($x_{0}$).Moving on, $p_{\theta}(x_{0:T})$ is a joint distribution of all the latents and is referred to as the reverse process.  Specifically Equation 1 states  that to get back original $p_{\theta}(x_{0})$ we should integrate over all latents to eliminate the from the joint distribution $p_{\theta}(x_{0:T})$ which is intractable . Additionally the reverse process is a Markov Chain with learnt transitions( Gaussian based ) starting at $p(x_{T}) = \mathcal{N}(x_{T};0,I)$ so can be defined as the following

$$

\begin{aligned}
p_{\theta}(x_{0:T}) = p(x_{T})\prod_{t = 1} ^T p_{\theta}(x_{t-1} \mid x_{t}) \quad \quad (1b)  \\ 
p_{\theta}(x_{t-1} \mid x_{t}) =\mathcal{ N}(x_{t-1} ;\; \mu_{\theta}( x_{t},t) \:,\: \Sigma_{\theta}( x_{t},t) \: ) \quad (1c)
\end{aligned}

$$

The key differentiator for a diffusion model to that of a latent is the forward process ( Diffusion ) which is a fixed Markov chain that repeatedly adds noise to the data/image. The forward process is analogous to brownian motion in the idea that the image gets noisier with each time step in the forward diffusion process due to the variance scheduling of $\beta_{1} \dots \beta_{T}$ .  Below in Equation 2a is the definition of the forward process .


$$

\begin{aligned}
q(x_{1:T} \mid x_{0}) = \prod_{t = 1} ^T q(x_{t} \mid x_{t-1}) \quad \quad (2a)  \\ 
q(x_{t} \mid x_{t-1}) =\mathcal{ N}(x_{t} ;\; \sqrt{1-\beta_{t}}\: x_{t-1} , \;\beta_{t}I ) \quad (2b)
\end{aligned}

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

- 1) states that it is the expectation taken with respect to destribution q(x). This is needed to take expectations when more than one variable involved.
- 2) states the notation used by the DDPM paper to represent joint probability distributions.
- 3) states the formula for Kullback-Leibler Divergence which measures the divergences between two probability distribution.

Having that sorted we can now move on to the following start:

$$

\begin{aligned}  \\ 
 \log(\: p_{\theta}(x_{0})\:)  & =  log(\: p_{\theta}(x_{0})) \cdot \int q(x_{1:T}) \: dx_{1:T} \\
 &= \int log(\: p_{\theta}(x_{0})) \cdot q(x_{1:T}) \: dx_{1:T}  \\  
 &= \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]  \qquad \qquad \\

\end{aligned} \qquad(3a)

$$


Using Conditional Probability we can write 

$$

 p_{\theta}(x_{0}) =  \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T} \mid x_{0})}

$$

and after substituting Back into (3) and then multiplying by $\frac{q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0})}$ we get

$$

\begin{aligned}  \\
 &= \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_{1:T} \mid x_{0})} \cdot \frac{q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0})} )\:]  \\ \\
 &=\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) \cdot q(x_{1:T} \mid x_{0})}{q(x_{1:T} \mid x_{0}) \cdot p_{\theta}(x_{1:T} \mid x_{0})} )\:] 
\end{aligned} \qquad (3b) 


$$

To Further simplify we can split Expectation into two terms


$$

\begin{aligned}  \\
 &=\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} +\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{q(x_{1:T} \mid x_{0}) }{p_{\theta}(x_{1:T} \mid x_{0})})\:] }_{D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0}))} \\ \\ 
 &=\underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} + D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0}))
\end{aligned} \qquad (3c)

$$


Here the Second term is The KL divergence and it has a property that : 

$$

\begin{aligned} \\
D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0})) \geq 0
\end{aligned} 

$$

This fact then can be used to relate $\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ with ELBO and KL divergence as follows by rearranging

$$

\begin{aligned} \\ \\
 \mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   - \:ELBO &=  \: D_{KL}(q(x_{1:T} \mid x_{0})\:  \mid  \mid  \: p_{\theta}(x_{1:T} \mid x_{0})) \:  \\ 
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   - \:ELBO &\geq 0 \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   &\geq ELBO \\
\end{aligned} \qquad (3d)

$$

The gap between $\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0:T})\:)\:]$ and ELBO is defined as the tightness of the bound. Additionally since the KL Divergence determines the divergence between the posterior (Q) and prior(P) distributions the tighter the bound the better it approximates between the two distributions. This can be done by Maximising the ELBO. However we can do a trick to instead minimise the bound by multiplying by a negative sign. This makes it easier in PyTorch to Train 

$$

\begin{aligned} \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\:\log(\:p_{\theta}(x_{0})\:)\:]   &\geq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: \log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{ELBO} \\ \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\:p_{\theta}(x_{0})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})})\:]}_{-ELBO} \\ \\
\end{aligned} \qquad (3e)

$$

This is the same as the Equation 3 in the Denoising Diffusion Probabilistic Model Paper. We can then use Equation 1b and 2a to write the parts of the ELBO as follows

$$

\begin{aligned} \\
\mathbb{E}_{q(x_{1:T} \mid x_{0})}[\: -\log(\:p_{\theta}(x_{0})\:)\:]   &\leq \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ \: -\log\left( \: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})} \right)\: \right]}_{ELBO} \\  \\

&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(  p(x_{T})\prod_{t = 1} ^T \frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right) \right] \\ \\ \\
&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p(x_{T})) - \sum_{t\geq 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right] = L
\end{aligned} \qquad (3f)
$$

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

\begin{aligned}
	L &=  \underbrace{\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ \: -\log\left( \: \frac{p_{\theta}(x_{0:T}) }{q(x_{1:T} \mid x_{0})} \right)\: \right]}_{ELBO} \\  
	&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p(x_{T})) - \sum_{t\geq 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})} \right]  \\  
	&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p(x_{T})) - \sum_{t > 1} \log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t} \mid x_{t-1})}- \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right]  \qquad (5a)\\ \\ 

	&\text{using Baye's} \to \quad q(x_{t} \mid x_{t-1}) = q(x_{t}\mid x_{t-1},x_{0}) = \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t},x_{0})}{q(x_{t-1},x_{0})} \\ \\
	&= \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t}\mid x_{0})\cdot q(x_{0})}{q(x_{t-1}\mid x_{0})\cdot q(x_{0})} = \frac{q(x_{t-1}\mid x_{t},x_{0}) \cdot q(x_{t}\mid x_{0})}{q(x_{t-1}\mid x_{0})} \\ \\
    &\text{Substituting in (5a) we can obtain} \\ \\ 
	
	&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p(x_{T})) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}\cdot \frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) })\right] - \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right]  \; (5b)\\ \\  
	&\text{Focusing on the summation } \\ \\
	
&\sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}\cdot \frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) })\right] = \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})}) \right] +\sum_{t > 1} \left[\log(\frac{q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0}) }) \right] \\ \\
	
\end{aligned}

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
	&\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log(  p(x_{T})) - \log(q(x_{1} \mid x_{0}))- \log\left(\frac{1}{q(x_{T} \mid x_{0})}\right) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})})\right] - \log\left( \frac{p_{\theta}(x_{0} \mid x_{1})}{ q(x_{1} \mid x_{0})} \right) \; \right] \\
&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(   \frac{p(x_{T})}{q(x_{T} \mid x_{0})} \right) - \sum_{t > 1} \left[\log(\frac{p_{\theta}(x_{t-1} \mid x_{t})}{ q(x_{t-1}\mid x_{t},x_{0})})\right] - \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \; \right]\\
&=\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(   \frac{p(x_{T})}{q(x_{T} \mid x_{0})} \right) \right] + \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[\sum_{t > 1} \left[\log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})})\right]\right] - \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[- \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \;\right] \\ 
\end{aligned} (5c)

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
	=\sum_{t > 1} \left[\int\int q(x_{t-1},x_{t}  \mid x_{0}) \cdot \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \; dx_{t-1} \:dx_{t}   \right]\\
\end{aligned}

$$

Using Chain Rule  for Conditional Probability 

$$

q(x_{t-1}, x_{t}\mid x_{0}) = q(x_{t-1} \mid x_{t} ,x_{0})\cdot q(x_{t} \mid x_{0})

$$

Substituting this Back

$$

\begin{aligned}
	&=\sum_{t > 1} \left[\int\int q(x_{t-1} \mid x_{t} ,x_{0})\cdot q(x_{t} \mid x_{0})  \cdot \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \; dx_{t-1} \:dx_{t}   \right]\\
	&= \sum_{t > 1} \left[\int  q(x_{t} \mid x_{0}) \left[\int q(x_{t-1} \mid x_{t} ,x_{0})  \cdot \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \; dx_{t-1}\right] \:dx_{t}   \right] \\
	&= \sum_{t > 1} \left[\int  q(x_{t} \mid x_{0}) \cdot\mathbb{E}_{q(x_{t-1} \mid x_{t} ,x_{0})}\left[  \log(\frac{ q(x_{t-1}\mid x_{t},x_{0})}{p_{\theta}(x_{t-1} \mid x_{t})}) \right] \:dx_{t}   \right]\\
	&=\sum_{t > 1} \left[\int  q(x_{t} \mid x_{0}) D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t}))  \:dx_{t}   \right]\\
	&=\sum_{t > 1} \mathbb{E}_{q(x_{t} \mid x_{0})}\left[  D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t})) \right]\\
	&=\mathbb{E}_{q(x_{t} \mid x_{0})}\left[\sum_{t > 1}  D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t})) \right]\\
	
\end{aligned} (5d)


$$

Moving On we can look at the first part of the expectation from 5c. We can apply a similar trick as we did previously

$$

\begin{aligned}
	\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[ -\log\left(   \frac{p(x_{T})}{q(x_{T} \mid x_{0})} \right) \right] & = \int\dots \int q(x_{1:T} \mid x_{0}) \cdot\log\left(   \frac{q(x_{T} \mid x_{0})}{p(x_{T})} \right)dx_{1}\dots dx_{T} \\
	&=\int q(x_{T} \mid x_{0}) \cdot\log\left(   \frac{q(x_{T} \mid x_{0})}{p(x_{T})} \right) dx_{T} \\
	&=\mathbb{E}_{q(x_{T} \mid x_{0})} \left[D_{KL}(q(x_{T} \mid x_{0}) \mid\mid p(x_{T})) \right] 
\end{aligned} (5e)

$$

And the Last Part of the Expectation in 5c can't be simplified to a KL divergence but the same trick can be applied to integrate out all latent variables the Expectation is taken w.r.t to get $\mathbb{E}_{q(x_{1} \mid x_{0})}\left[- \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \;\right]$ . 

Moreover, The reason I showcased this instead of jumping directly to the result showcased in the paper is to make it more intuitive as to why the KL divergences  show up. We can now write all the expectations together under the entire distribution q by re-integrating the missing variables  as they can be marginalised if needed by doing the same trick of integrating out. So 5c becomes the following when we combine each intermediate stage.


$$
\begin{aligned}
	&\mathbb{E}_{q(x_{T} \mid x_{0})} \left[D_{KL}(q(x_{T} \mid x_{0}) \mid\mid p(x_{T})) \right] + \mathbb{E}_{q(x_{t} \mid x_{0})}\left[\sum_{t > 1}  D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t})) \right] - \mathbb{E}_{q(x_{1} \mid x_{0})}\left[- \log\left( p_{\theta}(x_{0} \mid x_{1}) \right) \;\right] \\
	&= \mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[D_{KL}(q(x_{T} \mid x_{0}) \mid\mid p(x_{T})) + \sum_{t > 1}  D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t}))\; - \log\left( p_{\theta}(x_{0} \mid x_{1}) \right)\right]
\end{aligned} (5f)


$$


My reason to why this is important is that it is tractable(because we specifically condition on $x_{0}$ ) to train a model based on this result. This is because the minimising the individual KL divergences for a time step t(this can be done by choosing a random t) is efficient to do rather than all possible t's . Additionally, since we are working with Gaussian distributions we can easily minimise the KL divergences which i will showcase next. This results in less noisy estimates than those if we were to compute via monte-carlo estimates for the entire time steps.

Moving on we need to find the distribution of   $q(x_{t-1}\mid x_{t},x_{0})$. Here we can relate back to the Baye's rule in (5a)

$$

\begin{aligned}
	 &q(x_{t-1}\mid x_{t},x_{0})  =  q(x_{t-1} \mid x_{t}) = \frac{q(x_{t}\mid x_{t-1}) \cdot q(x_{t-1}\mid x_{0})}{q(x_{t}\mid x_{0})} \\ \\
	 
\end{aligned}


$$

using the results of 2b and 4c we can derive the result for a distribution. This derivation is one of the longest one's i worked through.I tried to not miss any detail below.


$$

\begin{aligned}
	q(x_{t-1}\mid x_{t},x_{0}) &= \frac{\mathcal{ N}(x_{t} ;\; \sqrt{1-\beta_{t}}\: x_{t-1} , \;\beta_{t}I ) \cdot \mathcal{ N}(x_{t-1} ;\; \sqrt{\bar{\alpha_{t-1}}}\: x_{0} , \;(1-\bar{\alpha_{t-1}})I )}{\mathcal{ N}(x_{t} ;\; \sqrt{\bar{\alpha_{t}}}\: x_{0} , \;(1-\bar{\alpha_{t}})I )}\\
\end{aligned}

$$

These multivariate Gaussian distributions are Isotropic, which means variance is same in all directions(The Covariance matrix is a scaled Identity Matrix ) . This is an important property as we can write an isotropic gaussian of "k" dimensions:

$$

\begin{aligned}
	&\mathcal{N}(x;\mathbf{\mu},\Sigma) = \frac{1}{ (2\pi)^{k/2} \mid \Sigma \mid ^{1/2} } \exp(-\frac{1}{2} \cdot(x-\mu)^T)\Sigma^{-1}(x-\mu))\\ \\
	&\text{using the following facts} \quad \Sigma = \sigma^2I,\quad \lvert \Sigma  \rvert = \sigma^{2k}  \quad \text{and} \quad\lvert \lvert x-\mu \rvert  \rvert ^{2} = (x-\mu)^T(x-\mu)  =  \sum_{i=0}^k(x_{i} -\mu_{i})^2\\ 
	&\mathcal{N}(x;\mathbf{\mu},\Sigma) = \frac{1}{\sqrt{ (2\pi\sigma^2)^k  }} \exp\left( -\frac{\lvert \lvert x-\mu \rvert  \rvert^2 }{2\sigma^2} \right) = \prod_{i=0}^{k} \frac{1}{\sqrt{ 2\pi \sigma^2 }}\exp\left( -\frac{(x_{i}-\mu_{i})^2}{2\sigma^2} \right) \\ \\
	&\text{ we can use this relation to just focus on writing it as a single guassian for simplicity } \\ \\
	&\mathcal{N}(x;\mathbf{\mu},\Sigma) =  \frac{1}{\sqrt{ 2\pi \sigma^2 }}\exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right) 

	
\end{aligned}

$$

Now we can write the following which gets a bit intense and long.

$$


\small{

\begin{aligned}

	q(x_{t-1}\mid x_{t},x_{0}) &= \underbrace{ \frac{1}{\sqrt{ \left( \frac{2\pi (1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{(1-\bar{\alpha_{t}})}  \right)^k }}}_{H_{1}}\exp\left(    -\frac{1}{2}\cdot \left[ \underbrace{ \frac{(x_{t}-\sqrt{ \alpha_{t} }x_{t-1})^2}{(1-\alpha_{t})} + \frac{(x_{t-1} - \sqrt{ \bar{\alpha_{t-1}} }x_{0})^2}{(1-\bar{\alpha_{t-1}})}-\frac{(x_{t}-\sqrt{ \bar{\alpha_{t}} }x_{0})^2}{(1-\bar{\alpha_{t}})}}_{H_{2}}\right]\right)  \qquad \text{recall: } 1-\alpha_{t} = \beta_{t}\\
	\text{Expanding } \\
	H_{2} &= \frac{x_{t}^2 + \alpha_{t}x_{t-1}^2 - 2\sqrt{ \alpha_{t} }x_{t}x_{t-1}}{(1-\alpha_{t})}+\frac{x_{t-1}^2+\bar{\alpha_{t-1}}x_{0}^2-2\sqrt{ \bar{\alpha_{t-1}} }x_{t-1}x_{0}}{(1-\bar{\alpha_{t-1}}) }-\frac{x_{t}^2+\bar{\alpha_{t}}x_{0}^2-2\sqrt{ \bar{\alpha}_{t} }x_{t}x_{0}}{(1-\bar{\alpha_{t}})}\\ \\
	\text{Grouping terms for } \\
	H_{2}&= \underbrace{ x_{t-1}^2\left( \frac{\alpha_{t}}{1-\alpha_{t}}+\frac{1}{1-\bar{\alpha_{t-1}}} \right)-2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }x_{t}}{1-\alpha_{t}}+\frac{\sqrt{ \bar{\alpha_{t-1}} }x_{0}}{1-\bar{\alpha_{t-1}}} \right)}_{A}+\underbrace{x_{t}^2\left( \frac{1}{1-\alpha_{t}}-\frac{1}{1-\bar{\alpha_{t}}} \right)+x_{0}^2\left( \frac{\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t-1}}}-\frac{\bar{\alpha_{t}}}{1-\bar{\alpha_{t}}} \right) -\frac{2\sqrt{ \bar{\alpha_{t}} }x_{t}x_{0}}{1-\bar{\alpha_{t}}}}_{B} \\ 
	A &= x_{t-1}^2\left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})+1-\alpha_{t}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})} \right)
	-2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}\right) \\
	&= x_{t-1}^2\left( \frac{\alpha_{t}-\alpha_{t}\bar{\alpha_{t-1}}+1-\alpha_{t}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})} \right)
	-2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}\right) \\
	&=x_{t-1}^2\left( \frac{1-\bar{\alpha_{t}}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})} \right)
	-2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}\right) \\
	H_{2} &= \underbrace{  \left( \frac{1-\bar{\alpha_{t}}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})} \right)}_{K_{1}}\left[\underbrace{x_{t-1}^2
	-2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}\right)\cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right)}_{K_{2}} + \underbrace{B\cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right)}_{K_{3}} \right]
	\\ \\
	\text{Completing the square}\\
	K_{2}&= x_{t-1}^2 -2x_{t-1}\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)\\
	&= \underbrace{ \left( x_{t-1} - \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)  \right)^2}_{K_{4}} \underbrace{-  \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)^2}_{K_{5}} \\
\end{aligned}}

$$


$$

{\small

\begin{aligned}
\text{Expanding K5} &\\
K_{5} &=-  \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)^2 \\
&= - \left( \frac{ \alpha_{t} (1-\bar{\alpha_{t-1}})^2 x_{t}^2+\bar{\alpha_{t-1}} (1-\alpha_{t})^2x_{0}^2 - 2(\sqrt{ \alpha_{t} \bar{\alpha_{t-1}} }(1-\alpha_{t})(1-\bar{\alpha_{t-1}})x_{t}x_{0})}{{(1-\bar{\alpha_{t}}})^2} \right) \\
\end{aligned}
}

$$


$$

\small{

\begin{aligned}
\text{Expanding K3} \\ 
K_{3}  &= \left[x_{t}^2\left( \frac{1}{1-\alpha_{t}}-\frac{1}{1-\bar{\alpha_{t}}} \right)+x_{0}^2\left( \frac{\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t-1}}}-\frac{\bar{\alpha_{t}}}{1-\bar{\alpha_{t}}} \right) -\frac{2\sqrt{ \bar{\alpha_{t}} }x_{t}x_{0}}{1-\bar{\alpha_{t}}}\right] \cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right) \\
&= \left[x_{t}^2\left( \frac{1-\bar{\alpha_{t}} -(1-\alpha_{t})}{(1-\alpha_{t})(1-\bar{\alpha_{t}})} \right)+x_{0}^2\left( \frac{\bar{\alpha_{t-1}}(1-\bar{\alpha_{t}})- \bar{\alpha_{t}}(1-\bar{\alpha_{t-1}})}{(1-\bar{\alpha_{t-1}})(1-\bar{\alpha_{t}})} \right) -\frac{2\sqrt{ \bar{\alpha_{t}} }x_{t}x_{0}}{1-\bar{\alpha_{t}}}\right]\cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right) \\
&=\left[x_{t}^2\left( \frac{\alpha_{t}-\bar{\alpha_{t}}}{(1-\alpha_{t})(1-\bar{\alpha_{t}})} \right) +x_{0}^2\left( \frac{\bar{\alpha_{t-1}}-\bar{\alpha_{t-1}}\bar{\alpha_{t}}- \bar{\alpha_{t}}+\bar{\alpha_{t}}\bar{\alpha_{t-1}}}{(1-\bar{\alpha_{t-1}})(1-\bar{\alpha_{t}})} \right) -\frac{2\sqrt{ \bar{\alpha_{t}} }x_{t}x_{0}}{1-\bar{\alpha_{t}}}\right]\cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right) \\
&=\left[x_{t}^2\left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})}{(1-\alpha_{t})(1-\bar{\alpha_{t}})} \right) +x_{0}^2\left( \frac{\bar{\alpha_{t-1}}- \bar{\alpha_{t}}}{(1-\bar{\alpha_{t-1}})(1-\bar{\alpha_{t}})} \right)-\frac{2\sqrt{ \bar{\alpha_{t}} }x_{t}x_{0}}{1-\bar{\alpha_{t}}}\right]\cdot \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right)\\
&=\left[x_{t}^2\left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})^2}{(1-\bar{\alpha_{t}})^2} \right) +x_{0}^2\left( \frac{(\bar{\alpha_{t-1}}- \bar{\alpha_{t}})(1-\alpha_{t})}{(1-\bar{\alpha_{t}})^2} \right)-\frac{2\sqrt{ \bar{\alpha_{t}} }(1-\alpha_{t})(1-\bar{\alpha_{t-1}})x_{t}x_{0}}{(1-\bar{\alpha_{t}})^2} \right]\\
&=\left[x_{t}^2\left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})^2}{(1-\bar{\alpha_{t}})^2} \right) +x_{0}^2\left( \frac{\bar{\alpha_{t-1}}(1- \alpha_{t})(1-\alpha_{t})}{(1-\bar{\alpha_{t}})^2} \right)-\frac{2\sqrt{ \bar{\alpha_{t}} }(1-\alpha_{t})(1-\bar{\alpha_{t-1}})x_{t}x_{0}}{(1-\bar{\alpha_{t}})^2} \right]\\
&=\left[x_{t}^2\left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})^2}{(1-\bar{\alpha_{t}})^2} \right) +x_{0}^2\left( \frac{\bar{\alpha_{t-1}}(1-\alpha_{t})^2}{(1-\bar{\alpha_{t}})^2} \right)-\frac{2\sqrt{ \bar{\alpha_{t}} }(1-\alpha_{t})(1-\bar{\alpha_{t-1}})x_{t}x_{0}}{(1-\bar{\alpha_{t}})^2} \right]


	


\end{aligned}
}

$$

The coefficients in K3 are same for K5 and hence $K_{3}+K_{5} = 0$ . Therefore we can write $H_{2}$ as follows:

$$

\small{
\begin{aligned}
	H_{2} &=  \left( \frac{1-\bar{\alpha_{t}}}{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})} \right)\left( x_{t-1} - \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)  \right)^2 \\
	&= \frac{1}{
	\left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right)
	}
	\left( x_{t-1} - \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)  \right)^2 \\
\end{aligned}}

$$

We can now write the following:

$$

\small{
\begin{aligned}
	q(x_{t-1}\mid x_{t},x_{0}) &=  \frac{1}{\sqrt{ \left( \frac{2\pi (1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{(1-\bar{\alpha_{t}})}  \right)^k }}\exp\left(     \frac{-1}{
	2\left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right)
	} \cdot
	\left( x_{t-1} - \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)  \right)^2 \right)
	
\end{aligned}
}

$$

We can now easily identify the variance and mean now:

$$

\begin{aligned}
	Variance &= \left( \frac{(1-\alpha_{t})(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right) = \left( \frac{\beta_{t}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_{t}}} \right) = \tilde{\beta_{t}} \\ 
	Mean &=\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(1-\alpha_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)\\
	&=\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(\beta_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right) = \tilde{\mu}_{t}(x_{t},x_{0})\\
	q(x_{t-1}\mid x_{t},x_{0}) &= \mathcal{N}(x_{t-1};\tilde{\mu}_{t}(x_{t},x_{0}),\tilde{\beta_{t}}I)
\end{aligned}

$$

### Unraveling the KL Divergences
In this section we revisit 

$$
\mathbb{E}_{q(x_{1:T} \mid x_{0})}\left[\underbrace{D_{KL}(q(x_{T} \mid x_{0}) \mid\mid p(x_{T}))}_{L_{T}} + \sum_{t > 1}  \underbrace{D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t}))}_{L_{t-1}}\; \underbrace{- \log\left( p_{\theta}(x_{0} \mid x_{1}) \right)}_{L_{0}}\right]
$$
 Here $L_{T}$ is essentially constant because $p(x_{T}) =\mathcal{N}(0,I)$ and doesn't depend on the parameters $\theta$.Therefore we can do no gradient updates.

Moving on we can now look at $L_{t-1}$ which can be used for training. Ideally we reverse process $p_{\theta}$ to have the same variance as the forward process to minimise the Divergence $p_{\theta}$ between reverse  and forward $q$  processes. Therefore we fix the reverses Variance $\Sigma_{\theta}( x_{t},t) \: =\sigma^2I=\tilde{\beta_{t}}I$. I used this relation when building my model, however the authors of DDPM suggest that $\Sigma_{\theta}( x_{t},t) \: =\sigma^2I=\beta_{t}I$ also works similarly. Moving on we need to look at the KL divergence relationship between multivariable Gaussians. The proof is a bit of a small tangent to write about here, so i suggest looking at the references. In the future I will possibly do a post documenting it. Below I state the KL divergence formula we are going to use.


$$

\begin{aligned}
 P: X \sim \mathcal{N}(\mu_{1},\Sigma_{1}) \\ 
 Q: X \sim \mathcal{N}(\mu_{2},\Sigma_{2}) \\ 
	D_{KL}(P\mid \mid Q) &= \frac{1}{2}\left[(\mu_{2}-\mu_{1})^T\Sigma^{-1}_{2}(\mu_{2}-\mu_{1})+tr(\Sigma^{-1}_{2}\Sigma_{1}) - \log\left( \frac{\lvert \Sigma_{1} \rvert }{\lvert \Sigma_{2} \rvert } \right) -n\right] \\
	
\end{aligned}


$$


We can now use this to find $D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t})$ 


$$

\begin{aligned}
	q(x_{t-1}\mid x_{t},x_{0}) = \mathcal{N}(x_{t-1};\tilde{\mu}_{t}(x_{t},x_{0}),\tilde{\beta_{t}}I) \\
    p_{\theta}(x_{t-1} \mid x_{t}) =\mathcal{ N}(x_{t-1} ;\; \mu_{\theta}( x_{t},t) \:,\: \Sigma_{\theta}( x_{t},t) \: ) \\
\end{aligned}

$$


$$

\small{
\begin{aligned}
&\\
	
    D_{KL}(q(x_{t-1}\mid x_{t},x_{0})\mid \mid p_{\theta}(x_{t-1} \mid x_{t}) &=\frac{1}{2}\left[ \frac{(\mu_{\theta}(x_{t},t)-\tilde{\mu}_{t}(x_{t},x_{0}))^T(\mu_{\theta}(x_{t},t)-\tilde{\mu}_{t}(x_{t},x_{0}))}{\sigma^2} +tr(\Sigma^{-1}_{1}\Sigma_{1}) - \log\left( \frac{\lvert \Sigma_{1} \rvert }{\lvert \Sigma_{1} \rvert } \right) -n\right] \quad \text{ using } \Sigma_{2} = \Sigma_{1} = \tilde{\beta}_{t}I \\
    &=\frac{1}{2}\left[ \frac{\lvert \lvert \mu_{\theta}(x_{t},t)-\tilde{\mu}_{t}(x_{t},x_{0}) \rvert  \rvert^2 }{\sigma^2} +n - \log\left(  1 \right) -n\right] \\
    &=\frac{1}{2}\left[ \frac{\lvert \lvert \mu_{\theta}(x_{t},t)-\tilde{\mu}_{t}(x_{t},x_{0}) \rvert  \rvert^2 }{\sigma^2}\right] \\ 


\end{aligned} }


$$

We can now substitute previous results for $\tilde{\mu_{t}}(x_{t},x_{0})$ and rearrange (4c) to get $x_{0} = \frac{1}{\sqrt{ \bar{\alpha_{t}} }}\left[x_{t}- (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon)\right]$. Additionally for a ideal $\mu_{\theta}(x_{t},t)$ we want it to be equal to $\tilde{\mu_{t}}(x_{t},x_{0})$. Using this we can do the following 

$$

\begin{aligned}
	\mu_{\theta}(x_{t},t) = \tilde{\mu_{t}}(x_{t},x_{0}) = \tilde{\mu_{t}}\left(x_{t}, \frac{1}{\sqrt{ \bar{\alpha_{t}} }}\left[x_{t}- (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon_{\theta})\right]\right)
\end{aligned}


$$

Here $\epsilon_{\theta}$ is the noise the model predicts for the reverse process and depends $\epsilon$,$x_{0},x_{t}$. Specifically this reparametrisation to handle the stochastic problem in backpropgation.

$$

\small{
\begin{aligned}
	\frac{1}{2}\left[ \frac{\lvert \lvert \mu_{\theta}(x_{t},t)-\tilde{\mu}_{t}(x_{t},x_{0} \rvert  \rvert^2 }{\sigma^2}\right] &= \frac{1}{2\sigma^2}\left[\left\lvert  \left\lvert  \tilde{\mu_{t}}\left(x_{t}, \frac{1}{\sqrt{ \bar{\alpha_{t}} }}\left[x_{t}- (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon_{\theta})\right]\right)-\tilde{\mu}_{t}(x_{t},\frac{1}{\sqrt{ \bar{\alpha_{t}} }}\left[x_{t}- (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon)\right])  \right\rvert   \right\rvert^2 \right] (6a) \\ \\

\text{since we can write }\tilde{\mu_{t}}(x_{t},x_{0}) &= \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}+\sqrt{ \bar{\alpha_{t-1}} }(\beta_{t})x_{0}}{{1-\bar{\alpha_{t}}}} \right)  \\

&=\left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}}{{1-\bar{\alpha_{t}}}} + \frac{\sqrt{ \bar{\alpha_{t-1}} }(\beta_{t})}{{1-\bar{\alpha_{t}}}}\cdot \frac{1}{\sqrt{ \bar{\alpha_{t}} }}\left[x_{t}- (\sqrt{  1-\bar{\alpha_{t}}} \cdot \:\epsilon)\right]  \right) \\

&= \left( \frac{\sqrt{ \alpha_{t} }(1-\bar{\alpha_{t-1}})x_{t}}{{1-\bar{\alpha_{t}}}} + \frac{\beta_{t}x_{t}}{(1-\bar{\alpha_{t}})\sqrt{ \alpha_{t} }}- \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }\sqrt{ \alpha_{t} }} \right)\\

&= \left( \frac{\alpha_{t}(1-\bar{\alpha_{t-1}})x_{t}}{(1-\bar{\alpha_{t}})\sqrt{ \alpha_{t} }} + \frac{\beta_{t}x_{t}}{(1-\bar{\alpha_{t}})\sqrt{ \alpha_{t} }}- \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }\sqrt{ \alpha_{t} }} \right) \\
&=\left( \frac{(\alpha_{t}-\bar{\alpha_{t}}-1+\alpha_{t})x_{t}}{(1-\bar{\alpha_{t}})\sqrt{ \alpha_{t} }} - \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }\sqrt{ \alpha_{t} }} \right)\\
&=\frac{-1}{\sqrt{ \alpha_{t} }} \left(x_{t}+ \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }}\right) \\ \\ 
\text{we can substitue into  6a now  } \\
&=\frac{1}{2\sigma^2}\left[\left\lvert  \left\lvert  \frac{-1}{\sqrt{ \alpha_{t} }} \left(x_{t}+ \frac{\beta_{t}\epsilon_{\theta}}{\sqrt{ 1-\bar{\alpha_{t}} }}\right) - \frac{-1}{\sqrt{ \alpha_{t} }} \left(x_{t}+ \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }}\right) \right\rvert   \right\rvert^2 \right] \\
&=\frac{1}{2\sigma^2}\left[\left\lvert  \left\lvert  \frac{-1}{\sqrt{ \alpha_{t} }} \left(x_{t} -x_{t} + \frac{\beta_{t}\epsilon_{\theta}}{\sqrt{ 1-\bar{\alpha_{t}} }}- \frac{\beta_{t}\epsilon}{\sqrt{ 1-\bar{\alpha_{t}} }}\right) \right\rvert   \right\rvert^2 \right] \\
&=\frac{1}{2\sigma^2}\left[\left\lvert  \left\lvert \frac{\beta_{t}}{\sqrt{ \alpha_{t} }\cdot\sqrt{ 1-\bar{\alpha_{t}} }}(\epsilon-\epsilon_{\theta})  \right\rvert   \right\rvert^2 \right] \\
&=\frac{\beta_{t}^2}{2\sigma^2 \alpha_{t} (1-\bar{\alpha}_{t})}\left\lvert  \left\lvert (\epsilon-\epsilon_{\theta})  \right\rvert   \right\rvert^2 
	
\end{aligned}}


$$

Since we there is an expectation enclosing the KL divergence we can write.Note since Expectation and  Summation are linear operators I have brought the Expectation inside the Summation here for $L_{t-1}$.

$$

\begin{aligned}
	\mathbb{E}_{q(x_{t} \mid x_{0})}[\frac{\beta_{t}^2}{2\sigma^2 \alpha_{t} (1-\bar{\alpha}_{t})}\left\lvert  \left\lvert (\epsilon-\epsilon_{\theta})  \right\rvert   \right\rvert^2] 
	&=\mathbb{E}_{x_{0},\epsilon}[\frac{\beta_{t}^2}{2\sigma^2 \alpha_{t} (1-\bar{\alpha}_{t})}\left\lvert  \left\lvert (\epsilon-\epsilon_{\theta})  \right\rvert   \right\rvert^2]
\end{aligned}

$$

Additionally, the above is just a result due to reparametrisation. Due to the expectation now being Independent of $\theta$ we can bring the Gradient operator($\nabla_{\theta}$) inside and drop the constant.

$$

\nabla_{\theta}(\left\lvert  \left\lvert (\epsilon-\epsilon_{\theta})  \right\rvert   \right\rvert^2)

$$

Now the final part is regarding the $L_{0}$.
