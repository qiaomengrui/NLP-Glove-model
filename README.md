# Glove模型
**模型目的：求出能表示出词与词之间的关系的词向量  
Glove是基于统计的模型  
Glove的整体思路：通过对词的共现计数矩阵进行降维，来得到词向量**
## 一、 模型原理
### 1. 共现矩阵  
共现矩阵的每一行分别代表每个词的词向量  
假设样本数据集：
1. I like deep learning.   
2. I like NLP.   
3. I enjoy flying.  

通过统计词的出现次数绘出V\*V共现次数矩阵，如下图  
![共现矩阵](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E5%85%B1%E7%8E%B0%E7%9F%A9%E9%98%B5.png)  
### 2. 目标函数
目标函数经过推导得到的最终结果如下图  
![目标函数](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E5%85%B1%E7%8E%B0%E7%9F%A9%E9%98%B5.png)  
参数解释：
* Xij：表示词j出现在中心词i的上下文的次数  
* Xi：表示任何词出现在中心词i的上下文的次数
* wi：表示i的词向量
* f(Xij)：表示权重函数
* bi：表示偏置
* V：表示词的总数（去重）
### 3. 权重函数
权重函数的设计满足以下三点：
* f(0)=0，如果两个词没有共同出现过，权重为0，不参与训练
* 两个词出现的次数多，权重不能变小
* 对于高频词并且包含信息极少的词，不能取过高的值  

根据这三个特点，构造了权重函数，最终结果如下图  
![权重函数](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E5%85%B1%E7%8E%B0%E7%9F%A9%E9%98%B5.png)  

当超参数xmax=100，a=0.75，权重函数的图像如下  
![给参数的权重函数图像](https://github.com/qiaomengrui/NLP-Glove-model/blob/master/pic/%E5%85%B1%E7%8E%B0%E7%9F%A9%E9%98%B5.png)  
## 二. Glove相关知识
### 1. SVD
SVD是一种降维方式
## 三. Glove和word2vec比较

