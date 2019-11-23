# 朴素贝叶斯


朴素贝叶斯算法是有监督的学习算法，解决的是分类问题，如客户是否流失、是否值得投资、信用等级评定等多分类问题。该算法的优点在于简单易懂、学习效率高、在某些领域的分类问题中能够与决策树、神经网络相媲美。但由于该算法以自变量之间的独立（条件特征独立）性和连续变量的正态性假设为前提，就会导致算法精度在某种程度上受影响。
## 朴素贝叶斯理论
朴素贝叶斯是贝叶斯决策理论的一部分，所以在讲述朴素贝叶斯之前有必要快速了解一下贝叶斯决策理论。
### 1 贝叶斯决策理论
假设现在我们有一个数据集，它由两类数据组成，数据分布如下图所示：
![](https://img-blog.csdn.net/20170817202948492?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
我们现在用p1(x,y)表示数据点(x,y)属于类别1(图中红色圆点表示的类别)的概率，用p2(x,y)表示数据点(x,y)属于类别2(图中蓝色三角形表示的类别)的概率，那么对于一个新数据点(x,y)，可以用下面的规则来判断它的类别：
+ 如果p1(x,y) > p2(x,y)，那么类别为1
+ 如果p1(x,y) < p2(x,y)，那么类别为2
### 2 条件概率
条件概率(Condittional probability)，就是指在事件B发生的情况下，事件A发生的概率，用P(A|B)来表示。
![](https://img-blog.csdn.net/20170817203257120?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


这就是条件概率计算公式。
### 3 全概率公式
![](https://img-blog.csdn.net/20170817204047993?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
这是全概率公式


![](https://img-blog.csdn.net/20170817204118030?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
这是条件概率的另一种写法
### 4 贝叶斯推断
对条件概率公式进行变形，可以得到如下形式：
![](https://img-blog.csdn.net/20170817204150864?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
我们把P(A)称为"先验概率"（Prior probability），即在B事件发生之前，我们对A事件概率的一个判断。


P(A|B)称为"后验概率"（Posterior probability），即在B事件发生之后，我们对A事件概率的重新评估。

P(B|A)/P(B)称为"可能性函数"（Likelyhood），这是一个调整因子，使得预估概率更接近真实概率。

所以，条件概率可以理解成下面的式子：

`1|后验概率　＝　先验概率 ｘ 调整因子`

这就是贝叶斯推断的含义。我们先预估一个"先验概率"，然后加入实验结果，看这个实验到底是增强还是削弱了"先验概率"，由此得到更接近事实的"后验概率"。

在这里，如果"可能性函数"P(B|A)/P(B)>1，意味着"先验概率"被增强，事件A的发生的可能性变大；如果"可能性函数"=1，意味着B事件无助于判断事件A的可能性；如果"可能性函数"<1，意味着"先验概率"被削弱，事件A的可能性变小。
### 5 朴素贝叶斯推断
贝叶斯和朴素贝叶斯的概念是不同的，区别就在于“朴素”二字，朴素贝叶斯对条件个概率分布做了条件独立性的假设。 比如下面的公式，假设有n个特征：

![](https://img-blog.csdn.net/20170817204458620?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

由于每个特征都是独立的，我们可以进一步拆分公式
![](https://img-blog.csdn.net/20171017163604596?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYzQwNjQ5NTc2Mg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**贝叶斯分类器的基本方法**：在统计资料的基础上，依据某些特征，计算各个类别的概率，从而实现分类。
## 朴素贝叶斯优缺点：
### 朴素贝叶斯推断的一些优点：
+ 生成式模型，通过计算概率来进行分类，可以用来处理多分类问题。
+ 对小规模的数据表现很好，适合多分类任务，适合增量式训练，算法也比较简单。
### 朴素贝叶斯推断的一些缺点：
+ 对输入数据的表达形式很敏感。
+ 由于朴素贝叶斯的“朴素”特点，所以会带来一些准确率上的损失。
+ 需要计算先验概率，分类决策存在错误率。
## 朴素贝叶斯改进之拉普拉斯平滑
利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，即计算p(w0|1)p(w1|1)p(w2|1)。如果其中有一个概率值为0，那么最后的成绩也为0。

如果新实例文本，包含这种概率为0的分词，那么最终的文本属于某个类别的概率也就是0了。显然，这样是不合理的，为了降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。这种做法就叫做拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。

除此之外，另外一个遇到的问题就是下溢出，这是由于太多很小的数相乘造成的。学过数学的人都知道，两个小数相乘，越乘越小，这样就造成了下溢出。在程序中，在相应小数位置进行四舍五入，计算结果可能就变成0了。为了解决这个问题，对乘积结果取自然对数。通过求对数可以避免下溢出或者浮点数舍入导致的错误。同时，采用自然对数进行处理不会有任何损失。
## 总结
+ 在训练朴素贝叶斯分类器之前，要处理好训练集。
+ 根据提取的分类特征将文本向量化，然后训练朴素贝叶斯分类器
+ 去高频词汇数量的不同，对结果也是有影响的的。
+ 拉普拉斯平滑对于改善朴素贝叶斯分类器的分类效果有着积极的作用。
