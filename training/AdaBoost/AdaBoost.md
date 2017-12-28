### AdaBoost
1. 优点: 泛化错误率低,易编码,可以应用在大部分分类器上,无参数调整
2. 缺点: 对离群点敏感
3. 使用数据类型: 数值型和标称型数据
### AdaBoost的一般流程
1. 收集数据: 可以使用任意方法
2. 准备数据: 依赖于所使用的弱分类器类型,这里使用的时单层决策树,这种分类器可以处理任何数据类型.
    当然也可以使用任意分类器作为弱分类器.作为弱分类器,简单分类器的效果更好
3. 分析数据: 可以使用任意方法
4. 训练算法: AdaBoost的大部分时间都用在训练上,分类器将多次在同一数据集上训练弱分类器
5. 测试算法: 计算分类器的错误率
6. 使用算法: 同SVM一样,AdaBoost预测两个类别中的一个.如果想把它应用到多个类别的场合,
    那么就要像多分类SVM中的做法一样多AdaBoost进行修改
### AdaBoost运行过程:
训练数据中的每个样本,并赋予其一个权重值,这些权重构成了向量D.一开始,这些权重都初始化成相等值
1. 在训练器上训练场一个弱分类器并计算该分类器的错误率
2. 在同一数据集上再次训练弱分类器
  在分类器的第二次训练当中,会重新调整每个样本的权重,其中第一次分队的样本的权重将会降低,而第一次分错的样本的权重将会提高
  为了从所有弱分类器中得到最终的分类结果,AdaBoost为每个分类器都分配了一个权重值alpha,这些alpha值时基于每个弱分类器的错误率进行计算的

**错误率$\epsilon$定义**

$\epsilon=\dfrac{未正确分类的样本数}{所有样本数}$

**权重值alpha**

$\alpha=\dfrac{1}{2}\ln\left(\frac{1-\epsilon}{\epsilon}\right)$

**权重向量D**

如果某个样本被正确分类,那么该样本的权重更改为:$D_{i}^{(i+1)}=\frac{D_{i}^{i}e^{-\alpha}}{\sum(D)}$
如果某个样本被错分,那么该样本的权重更改为: $D_{I}{(i+1)}=\frac{D_{i}^{(i)}e^{\alpha}}{\sum(D)}$

### 构建多函数来建立单层决策树代码
1. 第一个函数将用于测试是否有某个值小于或者大于正在测试的阈值
2. 第二个函数会在一个加权数据集中循环,并找到具有最低错误率的单层决策树
  **伪代码如下**

  	将最小错误率minError设为正无穷
  	对数据集中的每一个特征(第一层循环):
  		对每个步长(第二层循环):
  		对每个不等号(第三层循环):
  			建立一颗单层决策树并利用加权数据集对它进行测试
  			如果错误率低于minError,则将当前单层决策树设为最佳单层决策树
  	返回最佳单层决策树
### 完整AdaBoost算法实现

	对每次迭代:
		利用buildStump()函数找到最佳的单层决策树
		将最佳单层决策树加入到单层决策数组
		计算alpha
		计算新的权重向量D
		更新累计类别估计值
		如果错误率等于0.0,则退出循环

### 在一个难数据集上的AdaBoost应用
1. 收集数据: 提供的文本数据
2. 准备数据: 确保类别标签是+1和-1而非0和1
3. 分析数据: 手工检查数据
4. 训练算法: 在数据上,利用adaboostTrainDS()函数训练处一系列的分类器
5. 测试算法: 我们拥有两个数据集.在不采用随机抽样的方法下,我们就会对AdaBoost和Logistic回归的结果进行完全对等的比较
6. 使用算法: 观察该例子上的错误率
### 正确率、召回率
TP:真正例
FP:伪正例
FN:伪反例
TN:真反例
**正确率:** $\frac{TP}{(TP+FP)}$
**召回率:**	$\frac{TP}{(TP+FN)}$
