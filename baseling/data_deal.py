import numpy as np

"""
总体步骤：
数据集加载
构建word2id并去除低频词
构建共现矩阵
生成训练集 
给出 self.coocs(非0共现矩阵的位置) 和 self.labels(coocs对应的共现次数)
"""

class Wiki_Dataset():
    def __init__(self, min_count, window_size, file_name):
        """给出参数"""
        self.min_count = min_count
        self.window_size = window_size
        self.read_data(file_name)
        self.get_co_occur()

    def read_data(self, file_name):
        """数据集加载"""
        self.data = open(file_name).read()
        self.data = self.data.split()  # 用空格进行分词

        # 创建字典，词作为index，存储词出现的个数
        self.word2freq = {}
        for word in self.data:
            if self.word2freq.get(word) != None:  # 出现过的词词频+1
                self.word2freq[word] += 1
            else:  # 未出现过的词词频为1
                self.word2freq[word] = 1

        # 创建字典，用于过滤个数小于min_count的词，并对词进行编号
        self.word2id = {}
        for word in self.word2freq:
            if self.word2freq[word] < self.min_count:  # 忽略词频小于min_count的单词
                continue
            else:
                if self.word2id.get(word) == None:  # 存入未存过的词
                    self.word2id[word] = len(self.word2id)  # 对词编号

    def get_co_occur(self):
        """构建共现矩阵"""
        # vocab_size就是通常都V
        vocab_size = len(self.word2id)
        # 根据词表大小初始化全0共现矩阵
        comat = np.zeros((vocab_size, vocab_size))

        # 填写共现矩阵
        for i in range(len(self.data)):
            # 低频词不在word2id中，则忽略
            if self.word2id.get(self.data[i]) == None:
                continue
            # 找到中心词的id
            w_index = self.word2id[self.data[i]]
            # 找与中心词共现的词
            for j in range(i - self.window_size, i + self.window_size):
                if j >= 0 and j < len(self.data):
                    # 周围词为低频词则忽略, 自己与自己共现次数为0
                    if self.word2id.get(self.data[j]) == None or i == j:
                        continue
                    # 找到周围词的id
                    u_index = self.word2id[self.data[j]]
                    comat[w_index][u_index] += 1  # 共现次数+1

        # 将非零部分（x,y）提取出来
        self.coocs = np.transpose(np.nonzero(comat))
        # 建立列表，存入与（x,y）对应的共现次数
        self.labels = []
        for i in range(len(self.coocs)):
            self.labels.append(comat[self.coocs[i][0]][self.coocs[i][1]])
        # labels转为矩阵
        self.labels = np.array(self.labels)
        return self.coocs, self.labels

    def __len__(self):  # 求长度
        return len(self.coocs)

    def __getitem__(self, index):  # 根据index返回对应的数据
        return self.coocs[index], self.labels[index]