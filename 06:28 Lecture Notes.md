# Optimization Algorithms for Machine/Deep Learning

## 	Deterministic optimization methods

$minf(x)$,   $x\in X$

$f$: differentiable

$||\triangledown f(x)-\triangledown f(y)||\le L||x-y||$,   $\forall x, y\in X$

*e.g.* $f(x)=\frac{1}{2}X^\mathrm{T}AX-b^\mathrm{T}x$

​	$\triangledown f(x)=Ax-b$

​	$||\triangledown f(x)-\triangledown f(y)||=||A(x-y)||\le ||A||||x-y||$,   $L=||A||$

**Lemma**   $f(x)\le f(y)+<\triangledown f(y), x-y>+\frac{1}{2}||x-y||^2$

Proof:	Let $\phi(t)=f(y+t(x-y))$

​		$\triangledown\phi(t)=\triangledown f(y+t(x-y))^\mathrm{T}(x-y)$

​		$\phi(1)-\phi(0)=\int_0^1\triangledown\phi(t)dt$

​		$f(x)-f(y)=\int_0^1\triangledown f(y+t(x-y))^\mathrm{T}(x-y)dt$

​		$f(x)-f(y)-<\triangledown f(y), x-y>=\int_0^1\triangledown f(y+t(x-y))^\mathrm{T}(x-y)dt-\int_0^1\triangledown f(y)^\mathrm{T}(x-y)dt$

​		$=\int_0^1(\triangledown f(y+t(x-y))-\triangledown f(y))^\mathrm{T}(x-y)dt$

​		$\le\int_0^1||\triangledown f(y+t(x-y))||||x-y||dt$   (Cauchy inequality)

​		$\le tL||x-y||^2dt$

​		$=\frac{L}{2}||x-y||^2$

$x_{t+1}=argmin_{x\in X}Y_t<\triangledown f(x), x>+\frac{1}{2}||x-x_t||^2$

**Optimality condition for the above subproblem**

- $<\gamma_t\triangledown f(x_t)+x_{t+1}-x_t, x-x_{t+1}>\ge 0$,   $\forall x\in X$		(OPT1)
- $\gamma<\triangledown f(x_t), x_{t+1}-x>\le\frac{1}{2}||x-x_{t}||^2-\frac{1}{2}||x-x_{t+1}||-\frac{1}{2}||x_t-x_{t+1}||^2$		(OPT2)



### Observation

$f(x_{t+1})\le f(x_t)$ if $\gamma_t\le\frac{2}{L}$

Fix $x=x_t$ in OPT1, $<\triangledown f(x_t), x_{t+1}-x_t>\le-\frac{1}{\gamma_t}||x_{t+1}-x_t||^2$

Also, $f(x_{t+1})\le f(x_t)+<\triangledown f(x_t), x_{t+1}-x_t>+\frac{L}{2}||x_{t+1}-x_t||^2$			

​			$\le f(x_t)-(\frac{1}{\gamma_t}-\frac{L}{2})||x_{t+1}-x_t||^2$

​			$\le f(x_t)$

$f(x_{t+1})\le f(x_t)+<\triangledown f(x_t), x_{t+1}-x_t>+\frac{1}{2}||x_{t+1}-x||^2$

​		$=f(x_t)+<\triangledown f(x_t), x-x_t>+<\triangledown f(x_t), x_{t+1}-x>+\frac{L}{2}||x_{t+1}-x_t||^2$

​			(Strong) Convexity				OPT2

​		$\le f(x)+\frac{1}{2\gamma_t}[||x-x_t||^2-||x-x_{t+1}||^2]-\frac{1}{2}(\frac{1}{\gamma_t}-L)||x_{t+1}-x_t||^2$

​		$\le f(x)+\frac{1}{2\gamma_t}[||x-x_t||^2-||x-x_{t+1}||^2]$

If $\gamma_t=\gamma\le\frac{1}{L}$,   $t\le 1,2, …$, then $\sum_{t=1}^k[f(x_{t+1})-f(x)]\le \frac{1}{2\gamma}[||x-x_t||^2-||x-x_{t+1}||^2]$

Notice	$f(x_t)\ge f(x_{k+1})$,   $\forall t\le k+1$

​		$\sum_{t=1}^k[f(x_{t+1})-f(x)]\ge k[f(x_{k+1})-f(x)]$

Then 	$f(x_{k+1})-f(x)\le\frac{1}{2\gamma k}||x-x_1||^2$

​		$\gamma =\frac{1}{L}\Rightarrow f(x_{k+1})-f(x^*)\le\frac{L}{2k}||x^*-x_1||^2$



### Strong Convexity

$f(x)\ge f(y)+<\triangledown f(y), x-y>+\frac{\mu}{2}||x-y||^2$

*e.g.* $f(x)=\frac{1}{2}x^\mathrm{T}Ax+b^\mathrm{T}x$,   $\mu=\lambda_{min}(A)$

$f(x_{t+1})\le f(x)-\frac{\mu}{2}||x-x_t||^2+\frac{1}{2\gamma}[||x-x_t||^2-||x-x_{t+1}||^2]$

$f(x+1)-f(x^*)+\frac{1}{2\gamma}||x^*-x_{t+1}||^2\le \frac{1}{2}(\frac{1}{\gamma}-\mu)||x^*-x_t||^2$

$||x_{t+1}-x^*||\le (1-\gamma\mu)||x^*-x_t||^2$

If $r=\frac{1}{L}$, then $||x_{t+1}-x^*||\le (1-\frac{\mu}{L})||x_t-x^*||^2$

​			$||x_{k+1}-x^*||\le (1-\frac{\mu}{L})^k||x_t-x^*||^2$

If we want to have $||x_{k+1}-x^*||\le\varepsilon$

It suffices to have $(1-\frac{\mu}{L})^k||x_1-x^*||^2\le\varepsilon$

​				$(1-\frac{\mu}{L})^k\le \frac{\varepsilon}{||x_1-x^*||^2}$

​				$k\cdot log(1-\frac{\mu}{L})\le log\frac{\varepsilon}{||x_i-x^*||^2}$

​				$k\cdot(-log(1-\frac{\mu}{L}))\ge log\frac{||x_i-x^*||^2}{\varepsilon}$

​				$k\ge \frac{1}{-log(1-\frac{\mu}{L})}log\frac{||x_1-x^*||^2}{\varepsilon}\Leftarrow l\ge\frac{L}{\mu}log\frac{||x_i-x^*||^2}{\varepsilon}$

​				$\varepsilon$: conditional number

$\triangledown f(x)$

$E[G(x_t, \xi_t)]=\triangledown f(x_t)$

Define $\delta_t=\triangledown f(x_t)-G(x_t, \xi_t)$

- $E[\delta_t]=0$

  $\delta_t$ independent of $x_t$

- $E[||\delta_t||^2]\le\sigma^2$

$x_{t+1}=argmin_{x\in X}\gamma_t<G(x_t, \xi_t), x>+\frac{1}{2}||x-x_t||^2$

$\gamma_t<G(x_t, \xi_t), x_{t+1}-x>\le\frac{1}{2}[||x-x_t||^2-||x-x_{t+1}||^2-||x_t-x_{t+1}||^2]$   (OPT2')



`todo:`

Will be inplemented later or you can pull requests my [Github Repo](https://github.com/lcyyzy/Summer_Notes)



### Comments

1. $f(x)=E_{\xi}[F(x, \xi)]$, $\xi$ is continuous random variable, SGD nearly optimal
2. $f(x)=\frac{1}{N}\sum_{t=1}^Nf_i(x)$, using randomized incremental gradient method, we can improve the speed of convergence in terms of the dependence on $\varepsilon$. But the convergence depends on $N$.



- Deep Learning

- Burer-Monteiro Law Rank

  decomposition

  $X=LU$,   $L\in\mathbb{R}^{m\times r}$, $U\in\mathbb{R}^{r\times n}$

  $min_{L,U}||X-LU||^2$



### Nonconvex Optimization

$min_{x\in\mathbb{R}^n}f(x)$

$f$ is smooth but not necessarily convex

$||\triangledown f(x)-\triangledown f(y)||\le L||x-y||$,   $\forall x, y$

$x_{t+1}=x_t-\gamma_t\triangledown f(x_t)$

$f(x_{t+1})\le f(x_t)+\triangledown f(x_t)^\mathrm{T}(x_{t+1}-x_t)+\frac{L}{2}||x_{t+1}-x_t||^2$

​		$=f(x_1)-\gamma_t||\triangledown f(x_t)||^2+\frac{L\gamma_t^2}{2}||\triangledown f(x_t)||^2$

​		$=f(x_t)-\gamma_t(1-\frac{L\gamma_t}{2})||\triangledown f(x_t)||^2$

$\gamma_t(1-\frac{L\gamma_t}{2})||\triangledown f(x_t)||^2\le f(x_t)-f(x_{t+1})$

$\sum_{t=1}^k\gamma_t(1-\frac{L\gamma_t}{2})||\triangledown f(x_t)||^2\le f(x_1)-f(x_{t+1})\le f(x_1)-f^*$

Output $\overline{x_k}$ s.t. $||\triangledown f(\overline{x_k})||=min_{t=1, …, k}||\triangledown f(\overline{x_t})||$

$0\le \gamma_t \le \frac{2}{L}$

$\sum_{t=1}^k\gamma_t(1-\frac{L\gamma_t}{2})||\triangledown f(x_t)||^2\ge ||\triangledown f(\overline{x_k})||^2\sum_{t=1}^kr_t(1-\frac{Lr_t}{2})$



`todo:`

Will be inplemented later or you can pull requests my [Github Repo](https://github.com/lcyyzy/Summer_Notes)



$\sum d_i = n, d_i >= 1 $

















