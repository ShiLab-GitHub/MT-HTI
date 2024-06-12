import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
# 读取数据集
combined_df = pd.read_csv('../data/comparison/dataset.csv')

# 分割数据集为训练集、验证集和测试集，按照8:1:1的比例
train_data, test_data = train_test_split(combined_df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

# 加载训练好的模型
with open("metapath2vec_model4.pkl", "rb") as f:
    metapath2vec_embedding = pickle.load(f)

# 为节点对获取嵌入向量
def get_node_pair_embeddings(node1, node2, embedding_model):
    try:
        node1_emb = embedding_model.get_node_embedding(node1)
        node2_emb = embedding_model.get_node_embedding(node2)
        return np.concatenate([node1_emb, node2_emb])
    except KeyError:
        return np.zeros(embedding_model.vector_size * 2)

# 获得所有节点对的嵌入
train_embeddings = np.array([
    get_node_pair_embeddings(row['herb'], row['target'], metapath2vec_embedding)
    for index, row in train_data.iterrows()
])
val_embeddings = np.array([
    get_node_pair_embeddings(row['herb'], row['target'], metapath2vec_embedding)
    for index, row in val_data.iterrows()
])
test_embeddings = np.array([
    get_node_pair_embeddings(row['herb'], row['target'], metapath2vec_embedding)
    for index, row in test_data.iterrows()
])
# 将numpy数组转换为torch张量
train_embeddings_tensor = torch.tensor(train_embeddings)
val_embeddings_tensor = torch.tensor(val_embeddings)
test_embeddings_tensor = torch.tensor(test_embeddings)

# 将标签转换为torch张量
train_labels_tensor = torch.tensor(train_data['label'].values)
val_labels_tensor = torch.tensor(val_data['label'].values)
test_labels_tensor = torch.tensor(test_data['label'].values)

# 创建数据加载器
train_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_embeddings_tensor, val_labels_tensor)
test_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
# 评估函数
def evaluate_model(model, data_loader, val='val'):
    model.eval()
    predictions = []
    true_labels = []
    pre_lable1 = []
    with torch.no_grad():
        for batch in data_loader:
            embeddings, labels = batch
            outputs = model(embeddings)
            # 这里不需要再次使用 .squeeze(1)
            logits = outputs
            probabilities = torch.sigmoid(logits)  # 将 logits 转换为概率

            predicted = (probabilities > 0.5).long()  # 生成预测的标签
            pre_lable1.extend(predicted.cpu().numpy())
            # 假设 outputs 是模型的输出 Tensor
            predictions.extend(outputs.cpu().detach().numpy())
            true_labels.extend(labels.numpy())
    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)
    f1 = f1_score(true_labels, pre_lable1, average='macro')
    print(f"{val} Set Evaluation Metrics:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")


    return roc_auc, pr_auc, f1

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_classes=1):
        super(TransformerModel, self).__init__()
        # 定义两个线性层，分别处理两个节点的嵌入向量
        self.linear_herb = nn.Linear(embedding_dim, embedding_dim)
        self.linear_target = nn.Linear(embedding_dim, embedding_dim)

        # Transformer编码器层和编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim*2, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 添加 dropout 层
        self.dropout = nn.Dropout(0.1)  # 假设 dropout 率为 0.1

        # 分类头
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch_size, 2 * embedding_dim]
        # 分离两个节点的嵌入向量
        herb_emb = x[:, :embedding_dim]  # 第一个节点的嵌入向量
        target_emb = x[:, embedding_dim:]  # 第二个节点的嵌入向量

        # 通过各自的线性层
        herb_emb = self.linear_herb(herb_emb)
        target_emb = self.linear_target(target_emb)

        # 连接两个节点的嵌入向量
        combined_emb = torch.cat((herb_emb, target_emb), dim=1)

        # # 通过Transformer编码器之前添加 dropout 和 ReLU 激活函数
        # combined_emb = self.dropout(F.relu(combined_emb))

        # 通过Transformer编码器
        src = combined_emb.unsqueeze(0)  # Transformer期望序列的形式
        output = self.transformer_encoder(src)[0]  # 编码器输出

        # 取序列的第一个元素作为表示
        output = output.squeeze(0)  # [1, batch_size, embedding_dim] -> [batch_size, embedding_dim]

        # 通过分类头
        out = self.classifier(output)  # 直接输出 logits

        return out


class BinaryClassificationNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, dropout_rate=0.1):
        super(BinaryClassificationNetwork, self).__init__()
        # 第一个隐藏层
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        # 第二个隐藏层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        # 输出层，因为是二分类，所以使用sigmoid函数
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        herb_emb = x[:, :embedding_dim]  # 第一个节点的嵌入向量
        target_emb = x[:, embedding_dim:]  # 第二个节点的嵌入向量
        # 连接两个特征向量
        x = torch.cat((herb_emb, target_emb), dim=1)

        # 通过第一个隐藏层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用dropout
        x = self.dropout(x)

        # 通过第二个隐藏层和ReLU激活函数
        x = F.relu(self.fc2(x))
        # 应用dropout
        x = self.dropout(x)

        # 通过输出层
        out = self.output(x)

        # 使用sigmoid函数进行二分类预测
        return out
# 实例化模型
embedding_dim = 128 # 每个节点的嵌入维度
model = TransformerModel(embedding_dim)
patience = 40
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.005)
best_roc_auc = 0
# 训练模型
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_predictions = []
    train_true_labels = []
    pre_lable = []
    for batch in train_loader:
        embeddings, labels = batch
        optimizer.zero_grad()
        outputs = model(embeddings)

        # 这里不需要再次使用 .squeeze(1)
        logits = outputs
        probabilities = torch.sigmoid(logits)  # 将 logits 转换为概率

        predicted = (probabilities > 0.5).long()  # 生成预测的标签
        loss = criterion(logits.squeeze(1), labels.float())  # 计算损失
        loss.backward()
        optimizer.step()

        # 收集预测结果和真实标签
        train_predictions.extend(outputs.cpu().detach().numpy())
        train_true_labels.extend(labels.cpu().numpy())
        pre_lable.extend(predicted.cpu().numpy())

    # 计算 AUCROC、AUCPR 和 F1 分数
    train_roc_auc = roc_auc_score(train_true_labels, train_predictions)
    train_pr_auc = average_precision_score(train_true_labels, train_predictions)
    train_f1 = f1_score(train_true_labels, pre_lable, average='macro')  # 对于二分类问题使用 'binary'

    print(
        f'epoch: {epoch}, train_loss: {loss.item()}, train_aucroc: {train_roc_auc:.4f}, train_aucpr: {train_pr_auc:.4f}, train_f1: {train_f1:.4f}')
    # 在每个epoch后评估模型性能
    model.eval()
    with torch.no_grad():
        roc_auc, _, _ = evaluate_model(model, val_loader, 'Validation')
        # 早停逻辑
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break




# 在验证集和测试集上评估模型
val_roc_auc, val_pr_auc, val_f1 = evaluate_model(model, val_loader)
test_roc_auc, test_pr_auc, test_f1 = evaluate_model(model, test_loader, 'test')

# 保存所有结果到CSV文件
results_df = pd.DataFrame({
    'Data_Part': ['Validation', 'Test'],
    'ROC_AUC': [val_roc_auc, test_roc_auc],
    'PR_AUC': [val_pr_auc, test_pr_auc],
    'F1_Score': [val_f1, test_f1]

})
results_df.to_csv('evaluation_results.csv', index=False)