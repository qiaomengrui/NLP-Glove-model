# Glove模型
**Glove模型目的：求出能表示出词与词之间的关系的词向量   
Glove的整体思路：通过对词的共现计数矩阵进行降维，来得到词向量（共现：指语料中词汇一起出现的频率）**
## 一、 模型原理
模型实现：根据整个语料建立一个大型的体现词共现情况的矩阵，其目标是优化减小重建损失，即降维之后的向量能尽量表达原始向量的完整信息（共现矩阵+损失函数+权重函数）
### 1. 共现矩阵  
共现矩阵的概率比值可以用来区分词  
共现矩阵的每一行分别代表每个词的词向量  
假设样本数据集：
1. I like deep learning.   
2. I like NLP.   
3. I enjoy flying.  

通过统计词的出现次数绘出V\*V共现次数矩阵，如下图  
![共现矩阵](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E5%85%B1%E7%8E%B0%E7%9F%A9%E9%98%B5.png)  
缺点：  
当词汇量过大时，向量就会越大，矩阵会呈现更高的维度，同时需要更大的存储空间，而矩阵也会有稀疏的部分，这时就需要降维处理
### 2. 损失函数
损失函数经过推导得到的最终结果如下图  
![损失函数](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0.png)  
参数解释：
* Xij：表示词j出现在中心词i的上下文的次数  
* Xi：表示任何词出现在中心词i的上下文的次数
* wi：表示i的词向量
* f(Xij)：表示权重函数
* bi：表示偏置
* V：表示词的总数（去重）
### 3. 权重函数
权重函数相当于起降维的作用  
权重函数的设计需满足以下三点：
* f(0)=0，如果两个词没有共同出现过，权重为0，不参与训练
* 两个词出现的次数多，权重不能变小
* 对于高频词并且包含信息极少的词，不能取过高的值  

根据这三个特点，构造了权重函数，最终结果如下图  
![权重函数](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E6%9D%83%E9%87%8D%E5%87%BD%E6%95%B0.png)  

当超参数xmax=100，a=0.75，权重函数的图像如下  
![给参数的权重函数图像](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E7%BB%99%E5%8F%82%E6%95%B0%E7%9A%84%E6%9D%83%E9%87%8D%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png)  
## 二. Glove相关知识
### 1. SVD
SVD是一种降维方式  
但是SVD也有缺陷：  
1. 计算代价太大  
2. 难以将新的词合并进去  
### 2. 张量（Tensor）
张量是所有深度学习框架中最核心的组件，之后所有运算和优化算法都是基于张量进行的  
几何代数中定义的张量是基于向量和矩阵的推广，通俗一点理解的话，我们可以将标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量
## 三. Glove和word2vec比较
1. 整体上来看Glove相当于skip-gram的一种特殊形式，Glove的权重函数设置更加合理，所以训练效果更加好，权重函数就是skip-gram中对单词进行重采样处理  
2. 两个模型都可以根据词汇的共现矩阵，将词汇编码成一个向量  
3. word2vec基于预测的模型，其目标是不断提高对其他词的预测能力，即减小预测损失，从而得到词向量    
   Glove基于统计的模型，是通过对词的共现计数矩阵进行降维，来得到词向量。  
实际使用中，两种向量的对下游任务的效果并没有太大差别。
4. Glove相对于word2vec有一个优点是更容易并行化执行，可以更快，更容易地在大规模语料上训练
5. word2vec没有充分利用所有词，而Glove利用了所有词训练，所以效果要相比word2vec的训练效果好
## 四、Glove总结
Glove利用共现矩阵求得词向量，能够实现利用所有词对中心词的影响进行训练，总体来说相对其他的模型会更能表现出词与词之间细微的关系  
> Glove代码：https://github.com/qiaomengrui/NLP-Glove-model/tree/master/baseline  
> 论文：GloVe: Global Vectors for Word Representation
