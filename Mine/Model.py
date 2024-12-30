from pylab import *
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, APPNP, VGAE, GATConv
import sys
import torch
from torch_geometric.data import Data

# warnings.filterwarnings('ignore')
sys.path.append('./Data_Process')
path_result = "./Latent_representation/"


def process_adjTrain(data,num_drug, num_microbe):
    num_sum = num_drug + num_microbe
    team = csr_matrix((data), shape=(num_sum, num_sum))
    team = np.array(team[:(num_sum * num_sum), :num_sum].toarray() > 0, dtype=np.int)
    teams1 = team.nonzero()[1]
    teams0 = team.nonzero()[0]
    team = [teams0, teams1]
    data_tensor = torch.tensor(team)
    return data_tensor


def process_toTensor(data):
    a = np.array(data).conj().T
    processed_data = torch.tensor(a, dtype=torch.long)
    return processed_data


def GAAE(args_model, Adjacency_Matrix_raw, Features, train_adj, pos_test, neg_test, num_drug, num_microbe, pos_train, neg_train):

    train_adj = np.array(train_adj, copy=True)
    epochs = args_model.Epoch_Num
    features = Features
    data_tensor = process_adjTrain(train_adj, num_drug, num_microbe)
    all_pos_edges_index = process_adjTrain(Adjacency_Matrix_raw, num_drug, num_microbe)

    test_edges = process_toTensor(pos_test)
    test_edges_false = process_toTensor(neg_test)

    train_edges_false = process_toTensor(neg_train)

    data = Data(edge_index=data_tensor, x= features, test_mask=1373, train_mask=1373, val_mask=1373, y=1373)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index, train_edges_false)
        loss.backward()
        optimizer.step()
        return loss, z

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    def get_emb(pos_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, pos_edge_index)
        return z

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels, edge_index):
            super(Encoder, self).__init__()
            self.linear = nn.Linear(in_channels, out_channels)
            self.attention = GATConv(out_channels, out_channels, heads=1, dropout=0.4)
            self.propagate = APPNP(K=1, alpha=0)

        def forward(self, x, edge_index, not_prop=0):
            x = self.linear(x)
            x = F.leaky_relu(self.attention(x, edge_index))  # 使用自注意力机制
            x = self.propagate(x, edge_index)

            return x

    channels = args_model.Hidden_Layer_2


    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAE(Encoder(data.x.size()[1], channels, data_tensor)).to(dev)

    x, train_pos_edge_index = data.x.to(dev), data_tensor.to(dev)
    pos_edges_indecies = all_pos_edges_index.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args_model.Learning_Rate)  # 使用adam来做梯度下降

    max_auc = 0
    max_ap = 0
    max_acc, max_precision, max_recall, max_f1, max_mcc, best_threshold = 0, 0, 0, 0, 0, 0
    LOSS_Train = []
    LOSS_Test = []
    AUC = []
    for epoch in range(0, epochs):

        loss, emb = train()
        loss = float(loss)
        LOSS_Train.append(loss)

        with torch.no_grad():

            auc, ap, fpr, tpr, test_loss, acc, precision, recall, mcc, f1, threshold = test(test_edges, test_edges_false)
            LOSS_Test.append(test_loss)
            AUC.append(auc)

            if epoch % 10 == 0:
               print('------------------------------------------')
               print('Epoch: {:04d}, LOSS: {:.5f}'.format(epoch, loss))
               print('test:','AUC: {:.5f}, AP: {:.5f}, loss:{:.5f}'.format(auc, ap, test_loss))
            if auc > max_auc:
               max_auc = auc
               max_ap = ap
               max_acc, max_precision, max_recall, max_mcc, max_f1, best_threshold = acc, precision, recall, mcc, f1, threshold

    print('max_auc:', max_auc, 'ap:', max_ap,'acc:', max_acc,'precision:', max_precision,'recall:', max_recall,\
          'max_mcc:', max_mcc, 'f1:', max_f1, 'best_threshold:', best_threshold)
    #print(f"emb_shape:{emb.shape}")
    emb = get_emb(pos_edges_indecies)

    return max_auc, emb

