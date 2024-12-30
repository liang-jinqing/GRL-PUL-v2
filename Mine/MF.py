import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def WALS(R, K, max_iter, alpha=0.0002, lamda=0.002, observed_weight=1, unobserved_weight=0.1):
    M, N = R.shape

    # 初始化P和Q
    P = torch.rand(M, K, requires_grad=True)
    Q = torch.rand(N, K, requires_grad=True)

    for step in range(max_iter):
        predR = torch.mm(P, Q.t())
        loss = 0
        #loss = torch.sum((R - predR) ** 2)

        # 对于未观察到的评分，使用权重
        unobserved_mask = (R == 0)
        observed_mask = (R ==1)
        loss += torch.sum(unobserved_weight * (unobserved_mask * (R - predR)) ** 2)
        loss += torch.sum(observed_weight * (observed_mask * (R - predR)) ** 2)

        # 添加正则化项
        reg_loss = lamda * (torch.sum(P ** 2) + torch.sum(Q ** 2))
        loss += reg_loss

        if step % 1000 == 0:
            print(f"step:{step}, loss:{loss.item()}")

        # 计算梯度
        loss.backward()

        # 更新P和Q
        with torch.no_grad():
            P -= alpha * P.grad
            Q -= alpha * Q.grad

            # 清零梯度
            P.grad.zero_()
            Q.grad.zero_()

    return P, Q, loss.item()


def MF(R, K, max_iter, alpha, lamda, observed_weight, unobserved_weight):
    R = torch.tensor(R, dtype=torch.float32)
    P, Q, cost = WALS(R, K, max_iter, alpha, lamda, observed_weight, unobserved_weight)

    predR = np.dot(P.detach().numpy(), Q.detach().numpy().T)
    binary_predR = (predR >= 0.5).astype(int)

    auc = roc_auc_score(R.detach().numpy().flatten(), binary_predR.flatten())
    print(f"auc_MF:{auc}")

    return P, Q, binary_predR


def process_data_for_PU(emb_D, emb_M, predR):
    #num_drug = len(emb_D)
    #num_microbe = len(emb_M)
    num_drug, num_microbe = predR.shape[0], predR.shape[1]
    adj_MF = np.vstack((np.hstack((np.zeros(shape=(num_drug, num_drug), dtype=int), predR)),
                   np.hstack((predR.transpose(), np.zeros(shape=(num_microbe, num_microbe), dtype=int)))))
    embed = torch.cat((emb_D, emb_M), dim=0)

    return adj_MF, embed
