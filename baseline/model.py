import torch
import torch.nn as nn

class glove_model(nn.Module):
    def __init__(self, vocab_size, embed_size, x_max, alpha):
        """给参数和建矩阵"""
        super(glove_model, self).__init__()
        self.vocab_size = vocab_size  # 相当于V（去重的词个数）
        self.embed_size = embed_size  # 词向量的维度d
        self.x_max = x_max  # x_max是权重函数中上限,
        self.alpha = alpha  # alpha是权重函数中的指数，一般默认0.75

        # 待优化的矩阵 w_embed v_embed
        # 中心词向量，大小为V*d，转换为float64（double型）因为后面的label是double型的
        self.w_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64)
        # 中心词bias，每一个词对应一个bias
        self.w_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)
        # 周围词向量
        self.v_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64)
        # 周围词bias
        self.v_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)

    # w_data是batchsize的向量
    def forward(self, w_data, v_data, labels):
        """正向传播"""
        # embedding(tensor) 依次取出w_data的数字x，从w_embed中选取编号为x的向量放入w_data_embed中，把所有放入之后形成一个Tensor类型的变量
        w_data_embed = self.w_embed(w_data)  # batchsize*d
        w_data_bias = self.w_bias(w_data)  # batchsize*1
        v_data_embed = self.v_embed(v_data)  # batchsize*d
        v_data_bias = self.v_bias(v_data)  # batchsize*1

        # 生成权重，就是损失函数中的f(x_ij)
        weights = torch.pow(labels / self.x_max, self.alpha)
        weights[weights > 1] = 1

        # 这里最外面可以求和，为了值不要太大，我们用的是mean，对应的就是论文中的损失函数J的公式
        loss = torch.mean(weights * torch.pow(torch.sum(w_data_embed * v_data_embed, 1) + w_data_bias + v_data_bias - torch.log(labels), 2))
        return loss

    def save_embedding(self, word2id, file_name):
        """保存词向量"""
        # 这里和Word2Vec一样，用中心词和周围词的参数平均作为向量表示并保存
        embedding_1 = self.w_embed.weight.data.cpu().numpy()
        embedding_2 = self.v_embed.weight.data.cpu().numpy()
        embedding = (embedding_1 + embedding_2) / 2

        # 保存到文件
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(word2id), self.embed_size))
        for w, wid in word2id.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

