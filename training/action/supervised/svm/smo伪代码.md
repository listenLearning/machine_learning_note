**分割超平面:** $[Math Processing Error]w^Tx+b$

**点到分割面法线或垂线的长度:** $\frac{|w^TA+b|}{||w||}$

**空间向量:** $||w||=\sqrt[2]{x^2+y^2+z^2}$

**平面向量: ** $||w||=\sqrt[2]{x^2+y^2}$

**最大化最小间隔的数据点:** $arg\ max_{w,b}\{ (label\times(w^Tx+b))\times\frac{1}{||w||}\}$

**优化目标函数(拉格朗日乘子法):** $max_a[\sum_{i=1}^m\alpha-\frac{1}{2}\sum_{i,j=1}^m label^{(i)}\times label^{(j)}\times a_i \times a_j <x^{(i)},x^{(j)}>]$

**优化目标函数约束条件:** $c\geq \alpha \geq 0$ 并且 $ \sum_{i-1}^m \alpha \times label^{(i)} = 0$

**简化版本的smo伪代码**

	创建一个alpha向量并将其初始化为0向量
	当迭代次数小于最大迭代次数时(外循环)
	  对数据集中的每个数据向量(内循环):
	  	如果该数据向量可以被优化:
	      随机选择另外一个数据向量
	      同时优化这两个向量
	      如果这两个向量都不能被优化,退出内循环
	  如果所有向量都没有被优化,增加迭代数目,继续下一次循环
**径向基函数(高斯版本) ** $k(x,y)=\exp\left(\frac{-\parallel{x-y}\parallel^{2}}{2\sigma^{2}}\right) ,\\ \text{$\sigma$是用户定义的用于确定到达率(reach)或者说函数值跌落到0的速度参数}$

