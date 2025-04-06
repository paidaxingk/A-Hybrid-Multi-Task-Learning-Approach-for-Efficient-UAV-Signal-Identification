import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import f1_score
import time
import torchprofile
from torch.profiler import profile, ProfilerActivity
# import optuna
# from I_0_Multitask_AP_CLDNN import AP_ECA_CLDNN
# from I_1_1_Multitask_IQ_CLDNN import IQ_ECA_CLDNN

from model_mul.I_6_data import train_data,train_labels,val_data,val_labels,test_data,test_labels,train_yi_s,val_yi_s,test_yi_s
# from I_7open_RML2016_10a import train_data,train_labels,val_data,val_labels,test_data,test_labels,train_yi_s,val_yi_s,test_yi_s
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'train on:{device} ')
random.seed(60)
torch.manual_seed(90)
np.random.seed(100)
torch.cuda.manual_seed(60)
torch.cuda.manual_seed_all(42)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
file_path = "D:/Dataset/data_tensor_2048batch_size.pt"

# 将硬标签转换为软标签（标签平滑）
smoothing = 0.2  # 设置标签平滑参数
lr = 0.0004
epoch=400
patience=20

num_classes=5
y_smooth_train = train_labels * (1 - smoothing) + (smoothing / num_classes)
y_smooth_val = val_labels * (1 - smoothing) + (smoothing / num_classes)
y_smooth_test = test_labels * (1 - smoothing) + (smoothing / num_classes)


start_label = -20
label_step = 5
# 初始化标签列表
snr_labels = []

# 为每个样本打上信噪比标签
for i in range(9000):
    label = start_label + (i // 1000) * label_step
    snr_labels.append(label)
print(len(snr_labels))
# 转换为PyTorch张量
snr_labels = torch.tensor(snr_labels, dtype=torch.float32)


# 构建数据集
dataset_train = TensorDataset(train_data, train_labels,y_smooth_train,train_yi_s)
dataset_val = TensorDataset(val_data,val_labels,y_smooth_val,val_yi_s)
dataset_test = TensorDataset(test_data,test_labels,y_smooth_test,test_yi_s,snr_labels)
# 创建 DataLoader 对象
train_loader = DataLoader(dataset_train, batch_size=640)
val_loader = DataLoader(dataset_val, batch_size=640)
test_loader = DataLoader(dataset_test, batch_size=640)
# def data_loader(data, labels, y_smooth, yi_s, batch_size):
#     for i in range(0, len(data), batch_size):
#         yield data[i:i+batch_size], labels[i:i+batch_size], y_smooth[i:i+batch_size], yi_s[i:i+batch_size]
#
# def test_data_loader(data, labels, y_smooth, yi_s,snr_labels, batch_size):
#
#     for i in range(0, len(data), batch_size):
#         yield data[i:min(i+batch_size, len(data))], labels[i:min(i+batch_size, len(data))], y_smooth[i:min(i+batch_size, len(data))], yi_s[i:min(i+batch_size, len(data))], snr_labels[i:min(i+batch_size, len(data))]
# train_generator = data_loader(train_data, train_labels, y_smooth_train, train_yi_s, batch_size=128)
# val_generator = data_loader(val_data, val_labels, y_smooth_val, val_yi_s, batch_size=128)
# test_generator = test_data_loader(test_data, test_labels, y_smooth_test, test_yi_s,snr_labels,batch_size=128)











class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience  # 在多少个 epoch 之后停止
        self.min_delta = min_delta  # 验证集损失的最小提升
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 有改进，重置计数器
        else:
            self.counter += 1  # 无改进，计数器加1
            if self.counter >= self.patience:
                self.early_stop = True  # 达到 patience，触发提前停止

# train_loader = DataLoader(train_dataset,batch_size=640)
# val_loader = DataLoader(val_dataset,batch_size=640)
class eca_bloack(nn.Module):
    def __init__(self,channel,gamma = 2,b =1):
        super(eca_bloack, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+b)/ gamma))  # abs取绝对值
        kernel_size = kernel_size if kernel_size % 2 else kernel_size+1

        self.ave_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1,1,kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,l = x.size()
        avg = self.ave_pool(x).view([b,1,c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b,c,1])
        return out * x

class CompositeCrossEntropyLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(CompositeCrossEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target, smooth_target):
        # 计算传统交叉熵损失
        ce_loss = F.cross_entropy(output, target)

        # 计算平滑交叉熵损失
        # smooth_ce_loss = -torch.sum(smooth_target * F.log_softmax(output, dim=-1), dim=-1).mean()
        smooth_ce_loss = F.cross_entropy(output, smooth_target)

        # 计算复合损失
        comp_loss = self.alpha * ce_loss + (1 - self.alpha) * smooth_ce_loss

        return comp_loss

class IQ_ECA_CLDNN(nn.Module):
    def __init__(self,dropout_rate):
        super(IQ_ECA_CLDNN, self).__init__()
        self.CNN_layer = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8, stride=1, padding='same'),  # [batch_size, 64, 2048]
            nn.ReLU(),
            nn.MaxPool1d(2, 2),  # [batch_size, 64, 1024]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same'),  # [batch_size, 64, 1024]
            nn.ReLU(),  # [256, 64, 1024]
            nn.MaxPool1d(2, 2),  # [batch_size, 64, 512]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same'),  # [batch_size, 64, 512]
            nn.ReLU(),  # [256, 64, 512]
            nn.MaxPool1d(2, 2),  # [batch_size, 64, 256]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same'),  # [batch_size, 64, 256]
            nn.ReLU(),  # [256, 64, 256]
            nn.MaxPool1d(2, 2),  # [batch_size, 64, 128]
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, stride=1, padding='same'),  # [batch_size, 64, 128]
            nn.ReLU(),  # [256, 64, 128]
            nn.MaxPool1d(2, 2)  # [batch_size, 64, 64]
        )
        self.ECA_1d = eca_bloack(640)  # input_seq = (batch_size, seq_len, input_dim)
        # self.Multi_head_attention = nn.MultiheadAttention(64,8)
        self.lstm_layer_1 = nn.LSTM(64, 512, batch_first=True)# [batch_size, 64, 50]
        self.lstm_layer_2 = nn.LSTM(512, 512, batch_first=True)# [batch_size, 64, 50],取最后一个时间步的输出即[batch_size,50]
        self.DNN_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, input_tensor):
        model_output = self.CNN_layer(input_tensor)  # model_output shape:(128, 64, 64)-->(batch_size,64,64)
        lstm_layer_1_input = model_output.permute(0, 2 ,1)

        lstm_layer_1_output, _ = self.lstm_layer_1(lstm_layer_1_input)  # (64,128,50)
        lstm_layer_2_output, _ = self.lstm_layer_2(lstm_layer_1_output)
        # LSTM设置了batch_first,则输出(batch_size, seq_len, hidden_size)
        attention_input = lstm_layer_2_output[:, -1, :]  #(128, 50)
        attention_input = attention_input.unsqueeze(0)
        attention_output = self.ECA_1d(attention_input)

        DNN_input = attention_output.squeeze(0)
        DNN_output = self.DNN_layer(DNN_input)

        return DNN_input, DNN_output
class AP_ECA_CLDNN(IQ_ECA_CLDNN):
    def __init__(self,dropout_rate):
        super(AP_ECA_CLDNN, self).__init__(dropout_rate)

class Multitask_classifier(nn.Module):
    def __init__(self):
        super(Multitask_classifier,self).__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,P_assistant,P_H_AP,P_L_IQ):
        N = int(P_H_AP.shape[0])
        K = int(P_H_AP.shape[1])
        f_M = P_assistant[:, 0].unsqueeze(1) * P_L_IQ + P_assistant[:, 1].unsqueeze(1) * P_H_AP
        # print(P_assistant.shape,'\n',P_H_AP.shape,'\n',P_L_IQ.shape)
        # f_M = torch.zeros_like(P_H_AP)
        # for i in range(N):
        #     for j in range(K):
        #         f_M[i, j] = P_assistant[i, 1] * P_L_IQ[i, j] + P_assistant[i, 2] * P_H_AP[i, j]
        # f_M = P_assistant*(P_L_IQ+P_H_AP)
        P_M = self.softmax(f_M)
        # print(f'P_assistant:{P_assistant.shape},P_L_IQ.shape:{P_L_IQ.shape},P_H_AP.shape:{P_H_AP.shape},P_M.shape:{P_M.shape}')

        return P_M


class Assistant_DNN(nn.Module):
    def __init__(self,dropout_rate):
        super(Assistant_DNN,self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2),# 输出为[batch_size,5]
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out = self.DNN(x)
        return out



def visualize_features(features, labels):
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(np.unique(labels))):
        plt.scatter(features[np.array(labels) == i, 0], features[np.array(labels) == i, 1], c=colors[i],
                    label=str(i))
    plt.title('Data Visualization')
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.legend()
    plt.show()

#
# early_stopping = EarlyStopping(patience=3, min_delta=0)
alpha=0.5
composite_CELoss = CompositeCrossEntropyLoss(alpha=alpha)
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
Multitask_val_losses = []

def train(AP_model,IQ_model, Assistant_DNN,Multitask_classifier,train_loader, val_loader,
          optimizer,composite_CELoss, epochs):
    start_time = time.time()
    early_stop = EarlyStopping(patience)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        AP_model.train()
        IQ_model.train()
        Assistant_DNN.train()
        Multitask_classifier.train()
        total_running_loss = 0.0
        total_correct_train = 0
        total_train = 0

        for X_IQ, targets,labels_smooth,yi_s in train_loader:
            X_IQ=X_IQ.to(device)
            targets=targets.to(device)
            labels_smooth=labels_smooth.to(device)
            yi_s = yi_s.to(device)

            I = X_IQ[:, 0, :]
            Q = X_IQ[:, 1, :]
            phase = torch.atan2(Q, I)
            amplitude = torch.sqrt(I ** 2 + Q ** 2)
            X_AP = torch.cat((amplitude.unsqueeze(1), phase.unsqueeze(1)), dim=1)
            X_AP = X_AP.to(device)


            IQ_out_train, IQ_outputs = IQ_model(X_IQ)  # [batch,50],[batch,11]
            IQ_loss = composite_CELoss(IQ_outputs, targets, labels_smooth)
            # print(f'IQ_out_train:{IQ_out_train.shape},IQ_outputs.shape:{IQ_outputs.shape}')


            AP_out_train, AP_outputs = AP_model(X_AP)
            # print(f'X_AP:{X_AP.shape},AP_out_train:{AP_out_train.shape},AP_outputs{AP_outputs.shape}')
            AP_loss = composite_CELoss(AP_outputs, targets, labels_smooth)

            assistant_DNN_input = IQ_out_train + AP_out_train  # (batch,50)
            assistant_DNN_output = Assistant_DNN(assistant_DNN_input)  # [batch_size,2]
            # print(f'assistant_DNN_input:{assistant_DNN_input.shape},assistant_DNN_output:{assistant_DNN_output.shape}')
            # 其他辅助层输出为（batch_size,5）
            assistant_DNN_loss = F.cross_entropy(assistant_DNN_output, yi_s)
            #assistant_DNN_loss_detached = assistant_DNN_loss.detach()
            Multitask_classifier_output = Multitask_classifier(assistant_DNN_output,AP_outputs, IQ_outputs)  # (batch,5)
#               print(f'Multitask_classifier_output:{Multitask_classifier_output.shape}')
            Multitask_classifier_loss = composite_CELoss(Multitask_classifier_output, targets, labels_smooth)
            # 先不考虑中间辅助层梯度的影响。
            total_loss = AP_loss + IQ_loss + Multitask_classifier_loss+assistant_DNN_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_running_loss += total_loss.item() * X_IQ.size(0)
            _, predicted0 = torch.max(Multitask_classifier_output, 1)
            _, targets_indices = torch.max(targets, 1)
            total_correct_train += (predicted0 == targets_indices).sum().item()
            total_train += targets.size(0)
        epoch_train_loss = total_running_loss / len(train_loader.dataset)
        epoch_train_accuracy = total_correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Validation
        AP_model.eval()
        val_multitask_loss = 0.0
        val_corrected = 0
        total_val = 0
        AP_val_loss = 0.0
        AP_correct_val = 0
        AP_total_val = 0
        # 输出标签和预测，画混淆矩阵。
        # AP_all_targets = []
        # AP_all_predicted = []
        # AP_features = []

        IQ_model.eval()
        IQ_val_loss = 0.0
        IQ_correct_val = 0
        IQ_total_val = 0

        IQ_all_targets = []
        IQ_all_predicted = []
        IQ_features = []
        Assistant_DNN.eval()
        Multitask_classifier.eval()

        Multitask_val_accuracies = []
        total_running_loss_val = 0.0

        with torch.no_grad():

            for X_IQ, targets,labels_smooth,yi_s in val_loader:
                X_IQ = X_IQ.to(device)
                targets=targets.to(device)
                labels_smooth=labels_smooth.to(device)
                yi_s=yi_s.to(device)
                I = X_IQ[:, 0, :]
                Q = X_IQ[:, 1, :]
                phase = torch.atan2(Q, I)
                amplitude = torch.sqrt(I ** 2 + Q ** 2)
                X_AP = torch.cat((amplitude.unsqueeze(1), phase.unsqueeze(1)), dim=1)
                X_AP=X_AP.to(device)
                IQ_out_val, IQ_outputs_val = IQ_model(X_IQ)
#                print(f'input_noisy_signal_in_val:{input_noisy_signal_in_val.shape},IQ_out_val:{IQ_out_val.shape},IQ_outputs_val:{IQ_outputs_val.shape}')
                IQ_loss = composite_CELoss(IQ_outputs_val, targets, labels_smooth)
                IQ_val_loss += IQ_loss.item() * X_IQ.size(0)
                # _, IQ_predicted = torch.max(IQ_outputs_val, 1)  # torch.max(outputs, 1)：这个函数会返回张量 outputs 沿着第一个维度的最大值和最大值所在的索引。
                # IQ_correct_val += (IQ_predicted == targets).sum().item()
                # IQ_total_val += targets.size(0)
                #
                # IQ_all_targets.extend(targets.tolist())
                # IQ_all_predicted.extend(IQ_predicted.tolist())
                # IQ_features.extend(IQ_outputs_val.tolist())


                AP_out_val,outputs = AP_model(X_AP)
#                print(f'X_AP_val:{X_AP_val.shape},AP_out_val:{AP_out_val.shape},outputs:{outputs.shape}')
                AP_loss = composite_CELoss(outputs, targets,labels_smooth)
                AP_val_loss += AP_loss.item() * X_AP.size(0)
                # _, predicted1 = torch.max(outputs, 1)  # torch.max(outputs, 1)：这个函数会返回张量 outputs 沿着第一个维度的最大值和最大值所在的索引。
                # _, targets_indices = torch.max(targets, 1)
                # AP_correct_val += (predicted1 == targets_indices).sum().item()
                # AP_total_val += targets.size(0)

                # AP_all_targets.extend(targets.tolist())
                # AP_all_predicted.extend(predicted1.tolist())
                # AP_features.extend(outputs.tolist())
                val_assistant_DNN_input = IQ_out_val+AP_out_val  # [batch,50]
                val_assistant_DNN_output = Assistant_DNN(val_assistant_DNN_input)  # (batch,5)
#                print(f'val_assistant_DNN_input:{val_assistant_DNN_input.shape},val_assistant_DNN_output:{val_assistant_DNN_output.shape}')
                val_assistant_DNN_loss = F.cross_entropy(val_assistant_DNN_output,yi_s)
                val_Multitask_classifier_output = Multitask_classifier(val_assistant_DNN_output,outputs,IQ_outputs_val)
#                print(f'val_Multitask_classifier_output:{val_Multitask_classifier_output.shape}')
            # print(f'IQ_outputs:{IQ_outputs.shape},AP_outputs:{AP_outputs.shape},\nassistant_DNN_output:{assistant_DNN_output.shape}'
                     # f',Multitask_classifier_output:{Multitask_classifier_output.shape},\nIQ_outputs_val:{IQ_outputs_val.shape}'
                    # f'AP_outputs_val{outputs.shape},\nval_assistant_DNN_output:{val_assistant_DNN_output.shape},val_Multitask_classifier_output:{val_Multitask_classifier_output.shape}')
                val_Multitask_classifier_loss = composite_CELoss(val_Multitask_classifier_output,targets,labels_smooth)
                val_multitask_loss +=val_Multitask_classifier_loss.item()* X_AP.size(0)
                _,predicted3 = torch.max(val_Multitask_classifier_output,1)
                _, targets_indices = torch.max(targets, 1)
                val_corrected += (predicted3 == targets_indices).sum().item()
                total_val += targets.size(0)
                total_loss = IQ_loss+AP_loss+val_Multitask_classifier_loss+val_assistant_DNN_loss
                total_running_loss_val += total_loss.item()*X_AP.size(0)

            Multitask_epoch_val_loss = total_running_loss_val / len(val_loader.dataset)

            Multitask_epoch_val_accuracy = val_corrected / total_val
            Multitask_val_losses.append(Multitask_epoch_val_loss)
            Multitask_val_accuracies.append(Multitask_epoch_val_accuracy)
            val_accuracies.append(Multitask_epoch_val_accuracy)
        epoch_end_time = time.time()
        # epoch
        epoch_duration = epoch_end_time - epoch_start_time



        print(f"Multitask_model,Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {Multitask_epoch_val_loss:.4f},"
              f"Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {Multitask_epoch_val_accuracy:.4f}"
              f"completed in {epoch_duration:.2f} seconds")
        early_stop(Multitask_epoch_val_loss)
        if early_stop.early_stop:
            print("Early stopping triggered!")
            break
    end_time = time.time()
    # 总训练时长
    total_duration = end_time - start_time
    print(f'Total training time: {total_duration:.2f} seconds.')


        # early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        # for name, param in AP_ECA_CLDNN_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"AP_ECA_CLDNN_model,Layer: {name} | Gradients: {param.grad}")
        # for name, param in IQ_ECA_CLDNN_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"IQ_ECA_CLDNN_model,Layer: {name} | Gradients: {param.grad}")
        # for name, param in Assistant_DNN_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Assistant_DNN_model,Layer: {name} | Gradients: {param.grad}")
        # for name, param in Multitask_classifier_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Multitask_classifier_model,Layer: {name} | Gradients: {param.grad}")
    loss_data = {
        'train_loss': train_losses,
        'val_loss': Multitask_val_losses
    }

    # 使用 torch.save 将损失数据保存到文件
    torch.save(loss_data, 'D:/sunfengkui/Multitask_AssistantBackward/AssistantLossBackward_Multi-task_loss_data.pth')


# 用于计算 FLOP 的函数
def calculate_flops(model, input_data):
    model.eval()
    with torch.no_grad():
        flops = torchprofile.profile_macs(model, (input_data))
    return flops


results = {}
features_data ={}
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(AP_model,IQ_model,Assistant_DNN,Multitask_classifier,test_loader):
    total_flops = 0
    AP_model.to("cpu")
    IQ_model.to("cpu")
    Assistant_DNN.to("cpu")
    Multitask_classifier.to("cpu")

    AP_model.eval()
    IQ_model.eval()
    Assistant_DNN.eval()
    Multitask_classifier.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    features = []
    # 用于存储每个SNR值的样本数量和正确分类的样本数量
    samples_per_snr = {}  # 用来储存每个snr下的样本值
    correct_per_snr = {}  # 用来存储每个snr下的正确样本数量
    accuracy_per_snr = {}

    snr_targets = {}
    snr_predictions = {}
    snr_features = {}




    with torch.no_grad():


        for X_IQ, targets,labels_smooth,yi_s,snr_labels in test_loader:
            I = X_IQ[:, 0, :]
            Q = X_IQ[:, 1, :]
            phase = torch.atan2(Q, I)
            amplitude = torch.sqrt(I ** 2 + Q ** 2)
            X_AP = torch.cat((amplitude.unsqueeze(1), phase.unsqueeze(1)), dim=1)
            # X_AP=X_AP.to(device)

            AP_output1,AP_outputs = AP_model(X_AP)
            IQ_output1,IQ_outputs = IQ_model(X_IQ)
            assistant_DNN_input = AP_output1+IQ_output1
            assistant_DNN_output = Assistant_DNN(assistant_DNN_input)
            Multitask_classifier_out = Multitask_classifier(assistant_DNN_output,AP_outputs,IQ_outputs)




            AP_loss = composite_CELoss(AP_outputs,targets,labels_smooth)
            IQ_loss = composite_CELoss(IQ_outputs, targets, labels_smooth)
            # Assistant_DNN_loss = F.cross_entropy(assistant_DNN_output,yi_s).detach()
            Multitask_classifier_loss = composite_CELoss(Multitask_classifier_out,targets,labels_smooth)
            loss = AP_loss+IQ_loss+Multitask_classifier_loss
            test_loss += loss.item() * X_IQ.size(0)
            _, predicted = torch.max(Multitask_classifier_out, 1)  # 返回最大值，最大值所在索引
            _, targets_indices = torch.max(targets, 1)
            correct_test += (predicted == targets_indices).sum().item()
            total_test += targets.size(0)
            correct = (predicted == targets_indices).squeeze()
            for i, snr in enumerate(snr_labels):
                snr = snr.item()  # 获取snr值
                if snr not in samples_per_snr:  # 检查当前的SNR值是否已经存在于 samples_per_snr 字典中。如果不存在，说明这是第一次遇到这个SNR值。
                    samples_per_snr[snr] = 0  # 如果 samples_per_snr 字典中没有当前的SNR值，则将其初始化为0。这个字典用于记录每个SNR值的样本数量。
                    correct_per_snr[snr] = 0
                    snr_targets[snr] = []
                    snr_predictions[snr] = []
                    snr_features[snr] = []

                samples_per_snr[snr] += 1
                correct_per_snr[snr] += correct[i].item()
                snr_targets[snr].append(targets_indices[i])
                snr_predictions[snr].append(predicted[i])


                snr_features[snr].append(Multitask_classifier_out[i].numpy())

            AP_flops = calculate_flops(AP_model, X_AP)
            IQ_flops = calculate_flops(IQ_model, X_IQ)
            assistant_DNN_flops = calculate_flops(Assistant_DNN, assistant_DNN_input)
            Multitask_classifier_flops = calculate_flops(Multitask_classifier,
                                                         (assistant_DNN_output, AP_outputs, IQ_outputs))
            FLOP = AP_flops + IQ_flops + assistant_DNN_flops + Multitask_classifier_flops
            total_flops += FLOP


        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_accuracy = correct_test / total_test
        print(f"flops:{total_flops}MAC")

        AP_params = count_parameters(AP_model)
        IQ_params = count_parameters(IQ_model)
        Assistant_params = count_parameters(Assistant_DNN_model)
        Multitask_classifier_params = count_parameters(Multitask_classifier)
        print(f"Number of Parameters: {AP_params+IQ_params+Assistant_params+Multitask_classifier_params}")



    return epoch_test_accuracy


acc_list_Multi = []
train_losses_list = []
val_losses_list = []
train_accuracies_list = []
val_accuracies_list = []
dropout_rate = 0.5
AP_ECA_CLDNN_model = AP_ECA_CLDNN(dropout_rate).to(device)
IQ_ECA_CLDNN_model = IQ_ECA_CLDNN(dropout_rate=dropout_rate).to(device)
Assistant_DNN_model = Assistant_DNN(dropout_rate=dropout_rate).to(device)

Multitask_classifier_model = Multitask_classifier().to(device)
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
AP_ECA_CLDNN_model.apply(init_weights)
IQ_ECA_CLDNN_model.apply(init_weights)
Assistant_DNN_model.apply(init_weights)
Multitask_classifier_model.apply(init_weights)
optimizer = optim.RMSprop(list(AP_ECA_CLDNN_model.parameters())+
                              list(IQ_ECA_CLDNN_model.parameters())+
                              list(Multitask_classifier_model.parameters()),
                              lr=lr)
train_losses_list_in_test = []
val_losses_in_test = []
train_accuracies_in_test = []
val_accuracies_in_test = []
# train(AP_ECA_CLDNN_model,IQ_ECA_CLDNN_model,Assistant_DNN_model,Multitask_classifier_model,train_loader, val_loader,
#           optimizer, composite_CELoss,epochs=epoch)
train_losses_list.append(train_losses_list_in_test)
val_losses_list.append(val_losses_in_test)
train_accuracies_list.append(train_accuracies_in_test)
val_accuracies_list.append(val_accuracies_in_test)
test_acc = test(AP_ECA_CLDNN_model,IQ_ECA_CLDNN_model,Assistant_DNN_model,Multitask_classifier_model,test_loader)
acc_list_Multi.append(test_acc)



