import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from data_deal import Wiki_Dataset
from model import glove_model

# 超参数
settings = {
    'window_size': 2,  # 窗口尺寸 m
    'n': 10,  # 单词嵌入(word embedding)的维度,维度也是隐藏层的大小。
    'epochs': 50,  # 表示遍历整个样本的次数。在每个epoch中，我们循环通过一遍训练集的样本。
    'learning_rate': 0.001,  # 学习率
    'min_count': 0,  # 低于min_count去除
    'x_max': 100,  # 权重函数的x_max
    'a': 0.75,  # 权重函数a
    'epoch': 2,  # 迭代次数
    'batch_size': 5  # 一批次训练多少个数据
}

if __name__ == "__main__":
    # 数据预处理
    wiki_dataset = Wiki_Dataset(settings['min_count'], settings['window_size'], "./words.txt")
    # 模型初始化
    model = glove_model(len(wiki_dataset.word2id), settings['n'], settings['x_max'], settings['a'])
    # 实现优化
    optimizer = optim.Adam(model.parameters(), lr=settings['learning_rate'])
    # 损失值
    loss = -1
    # batch_size 一次性放入训练的个数
    training_iter = torch.utils.data.DataLoader(dataset=wiki_dataset, batch_size=settings['batch_size'], shuffle=True)

    # 进行迭代
    for epoch in range(settings['epoch']):
        # 随机在 training_iter 中选取 coocs 和对应的 labels
        for data, label in training_iter:
            # 将data的所有第0列放入w_data中，data所有的第1列放入v_data中，输出都是tensor类型
            # 不能直接给值得原因是：直接给值每个位置的类型是 tensor(x),整体是list类型。
                                # 下面方式给值是 [x]，然后整体式tensor类型
            w_data = torch.Tensor(np.array([sample[0] for sample in data])).long()
            v_data = torch.Tensor(np.array([sample[1] for sample in data])).long()

            # 正向传播
            # loss_now是tensor类型，需要取出当中值
            loss_now = model.forward(w_data, v_data, label)
            if loss == -1:  # 对损失进行平滑处理
                loss = loss_now.data.item()
            else:
                loss = 0.95 * loss + 0.05 * loss_now.data.item()
            optimizer.zero_grad()  # 清空过往梯度
            loss_now.backward()  # 反向传播，计算当前梯度；
            # 根据梯度更新网络参数(会使优化器迭代它应该更新的所有参数(张量),并使用它们内部存储的grad来更新它们的值)
            optimizer.step()
        print("loss: %s" % loss)

    # 保存词向量
    model.save_embedding(wiki_dataset.word2id, "./word_vectors.txt")