### FP-growth发现频繁项集的基本过程

    1.构建FP树
    2.从FP树中挖掘频繁项集
### FP树: 用于编码数据集的有效方式

    FP-growth算法
    优点: 一般要快于Apriori(Apriro算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁,而FP-growth算法只需要对数据进行两次扫描)
    缺点:实现比较困难,再某些数据集上性能会下降
    适用数据类型:标称型数据

FP-growth算法将数据存储在一种称为FP树的紧凑数据结构中.FP代表频繁模式(Frequent Pattern).一颗FP树看上去与计算机科学中的其他树结构类似,但是它通过链接(link)来连接相似元素,被连起来的元素项可以看成一个链表\
同搜索树不同的是,一个元素可以再一颗FP树中出现多次.FP树会存储项集的出现频率,而每个项集会议路径的方式存储再树中.存在相似元素的集合会共享树的一部分,只有当集合之间完全不同时,树才会分叉.树节点上给出集合中的单个元素及其再序列中的出现次数,路径就会给出该序列的出现次数.\
相似项之间的链接即**节点链接(node link)**,用于快速发现相似项的位置
### FP-growth算法的工作流程
- 首先构建FP树,然后利用它来挖掘频繁项集,为构建FP树,需要对原始数据集扫描两遍.第一遍对所有元素项的出现次数进行计数.如果某元素是不频繁的,那么包含该元素的超集也是不频繁的,所以就不需要考虑这些超集.数据库的第一遍扫描用来统计出现的频率,而第二遍扫描中只考虑那些频繁元素
### FP-growth一般流程

    1.收集数据: 使用任意方法
    2.准备数据: 由于存储的是集合,所以需要离散数据.如果要处理连续数据,需要将它们量化为离散值
    3.分析数据: 使用任意方法
    4.训练算法: 构建一个FP树,并对树进行挖掘
    5.测试算法: 没有测试过程
    6.使用算法: 可用于识别经常出现的元素项,从而用于制定决策、推荐元素或进行预测等应用
### 构建FP树
- **除了FP树之外,还需要一个头指针表来指向给定类型的第一个实例.利用指针表,可以快速访问FP树中一个给定类型的所有元素**    
- **除了存放指针外,头指针表还可以用来保存FP树中每类元素的总数**
- 第一次遍历数据集会获得每个元素项的出现频率.接下来,去掉不满足最小支持度的元素项.再下一步构建FP树.在构建时,读入每个项集并将其添加到一条已经存在的路径中.如果该路径不存在,则创建一条新路径.每个事务就是一个无序集合.假设有集合{z,x,y}和{y,z,r},那么在FP树中,相同项只会表示一次.为了解决此问题,在将集合添加到树之前,需要对每个集合进行排序.排序基于元素项的绝对出现频率来进行
- 在对事务记录过滤和排序之后,就可以构建FP树了.从空集开始,向其中不断添加频繁项集.过滤,排序后的事务依次添加到树中,如果树中已存在现有元素,则增加现有元素的值;如果现有元素不存在,则向树添加一个分枝.
### 从一颗FP树中挖掘频繁项集
- 有了FP树之后,就可以抽取频繁项集了,首先从单元素项集合开始,然后在此基础上逐步构建更大的集合.
- 从FP树中抽取频繁项集的三个基本步骤如下:\
    1.从FP树中获得条件模式基\
    2.利用条件模式基,构建一个条件FP树\
    3.迭代重复步骤1步骤2,直到树包含一个元素项为止
### 抽取条件模式基
- 首先从已经保存在头指针表中的单个频繁元素项开始.对于每一个元素项,获得其对应的**条件模式基(conditional pattern base)**.条件模式基是以所查找元素项为结尾的路径集合.每一条路径其实都是一条**前缀路径(prefix path)**.简而言之,一条前缀路径是介于所查找元素项与树根节点之间的所有内容
- 每一条前缀路径都与一个计数值关联.该计数值等于其实元素项的计数值
- 为了获得前缀路径,可以对数进行穷举式搜索,直到获得想要的频繁项为止.也可以利用头指针表来获得一种更有效的方法,头指针表包含相同类型元素链表的其实指针,一旦到达了每一个元素项,就可以上溯这颗树直到更节点为止.