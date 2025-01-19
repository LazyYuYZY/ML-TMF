import torch
import torch.nn as nn
import torch.optim as optim


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)  # 添加时间步维度
        embedded = self.embedding(trg)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc1(output.squeeze(1))
        prediction = self.fc2(output)
        return prediction, hidden, cell


# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[1]
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.fc2.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs


# 模型参数
INPUT_DIM = 1000  # 输入词表大小
OUTPUT_DIM = 1000  # 输出词表大小
ENC_EMB_DIM = 256  # 编码器嵌入维度
DEC_EMB_DIM = 256  # 解码器嵌入维度
HIDDEN_DIM = 512  # LSTM隐层维度
N_LAYERS = 2  # LSTM层数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# 模拟训练过程
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 示例数据迭代器 (这里只是一个示例，实际使用时应使用DataLoader)
class DummyIterator:
    def __iter__(self):
        return iter([(
            torch.randint(0, INPUT_DIM, (32, 10)),  # src
            torch.randint(0, OUTPUT_DIM, (32, 10))  # trg
        )])

    def __len__(self):
        return 100