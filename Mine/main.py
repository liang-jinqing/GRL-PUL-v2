import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef
from MF import MF, process_data_for_PU
from inits import load_data
import argparse
import torch
from Model import GAAE
from sklearn.ensemble import RandomForestClassifier
from puAdapter import PUAdapter
from tools import rondom_split, cross_valid_experiment, caculate_avg, process_emb
from utils_pu import sel_neg_by_bagging


def main():
    # ------------------data_parameters-----------------#
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")], default=('net1'),help="lists of graphs to use.")
    parser.add_argument('--attributes', type=lambda s: [item for item in s.split(",")], default=('features,similarity'), help=" attributes ： features similarity")
    parser.add_argument('--data_path', type=str, default='MDAD', help="lists of dataset : MDAD, DrugVirus, aBiofilm")
    parser.add_argument('--k_folds', type=int, default='5', help="Cross verify the added isolated nodes")
    args_data = parser.parse_args()
    k = args_data.k_folds
    print(args_data)
    graph = 'net1'
    adj, Features, A, labels, num_drug, num_microbe = load_data(graph, args_data)
    Features = torch.tensor(Features.toarray(), dtype=torch.float32)

    # -----------------MFPUB_module_parameters-----------------#
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MF_iteration', type=int, default='8000', help="")  # 8000
    parser.add_argument('--latent_factors_size', type=int, default='50', help="")
    parser.add_argument('--alpha', type=float, default='0.0002', help=" ")
    parser.add_argument('--lamda', type=float, default='0.006', help="")
    parser.add_argument('--observed_weight', type=int, default='1', help="")
    parser.add_argument('--unobserved_weight', type=float, default='0.1', help="")
    parser.add_argument('--PUB_iteration', type=int, default='20', help="")  # 20
    args_MFPUB = parser.parse_args()

    # -----------------GAAE_module_parameters-----------------#
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Epoch_Num', type=int, default='100', help="")
    parser.add_argument('--Learning_Rate', type=float, default='5e-4', help="")
    parser.add_argument('--Hidden_Layer_1', type=int, default='3092', help="")
    parser.add_argument('--Hidden_Layer_2', type=int, default='256', help="")
    args_GAAE = parser.parse_args()

    
    M_auc_GAAE, M_Tree_H = [], []
    M_auc_PU, M_aupr_PU, M_acc_PU, M_precision_PU, M_recall_PU, M_mcc_PU, M_f1_PU = [], [], [], [], [], [], []
    for itera in range(1):
        print("This is " + str(itera+1) +"-iteration")

        # ---------------------k-fold_valid_experiment-----------------#
        pos_test, neg_test, pos_test_Bip, neg_test_Bip, pos_pairs_all, unlabeled_pairs_all = cross_valid_experiment(adj, k, num_drug, num_microbe)
        emb_M, emb_D, predR = MF(A, K=args_MFPUB.latent_factors_size, max_iter=args_MFPUB.MF_iteration, alpha=args_MFPUB.alpha,\
                                 lamda=args_MFPUB.lamda, observed_weight=args_MFPUB.observed_weight, unobserved_weight=args_MFPUB.unobserved_weight)
        adj_MF, embed = process_data_for_PU(emb_M, emb_D, predR)
        neg_sel = sel_neg_by_bagging(pos_pairs_all, unlabeled_pairs_all, embed, adj_MF, iterate_time=args_MFPUB.PUB_iteration)
        neg_test = rondom_split(k, neg_sel)
        threshold = 0.5

        Results_GNAEMDA = []
        Results_PU = []
        Tree_H = []
        N_acc_PU, N_ap_PU, N_precision_PU, N_recall_PU, N_mcc_PU, N_f1_PU = [], [], [], [], [], []

        for i in range(k):
            print('----------------this is',i+1,'th corss---------------')

            train_adj = np.array(adj, copy=True)
            test_edge = pos_test[i]
            test_false = neg_test[i]

            train_edge = pos_test[:i] + pos_test[i + 1:]
            train_neg = neg_test[:i] + neg_test[i + 1:]
            train_edges = [list(item) for sublist in train_edge for item in sublist]
            train_negs = [list(item) for sublist in train_neg for item in sublist]

            train_edges_PU = pos_test[:i] + pos_test[i + 1:]
            train_negs_PU = neg_test[:i] + neg_test[i + 1:]

            adj_temp = np.array(adj, copy=True)
            for index in test_edge:
                adj_temp[index[0]][index[1]] = 0
                adj_temp[index[1]][index[0]] = 0

            for index in test_edge:
                train_adj[index[0]][index[1]] = 0
                train_adj[index[1]][index[0]] = 0


            print('--------GAAE module--------')
            result_GNAE, emb = GAAE(args_GAAE, adj, Features, adj_temp, test_edge, test_false, num_drug, num_microbe, train_edges, train_negs)
            Results_GNAEMDA.append(result_GNAE)

            train_edge_list = [list(item) for sublist in train_edges_PU for item in sublist]
            train_negs_list = [list(item) for sublist in train_negs_PU for item in sublist]

            train_index = train_edge_list + train_negs_list
            np.random.seed(i)
            np.random.shuffle(train_index)

            test_index = np.concatenate((test_edge, test_false), axis=0)
            np.random.seed(i)
            np.random.shuffle(test_index)

            X_train, Y_train = process_emb(emb, adj_temp, train_index)
            X_train, Y_train = np.array(X_train), np.array(Y_train)
            X_test, Y_test = process_emb(emb, adj, test_index)
            X_test, Y_test = np.array(X_test), np.array(Y_test)

            classifier = RandomForestClassifier(n_estimators=100)
            pu_estimator = PUAdapter(classifier)
            pu_estimator.fit(X_train, Y_train)



            tree_H = []
            #print("**************************************************")
            for i, tree in enumerate(classifier.estimators_):
                tree_height = tree.tree_.max_depth
                #print(f"Tree {i + 1} height: {tree_height}")
                tree_H.append(int(tree_height))
            #print("**************************************************")

            y_score = pu_estimator.predict(X_test)

            y_pred = [int(item > threshold) for item in y_score.flatten()]
            auc_pu = roc_auc_score(Y_test, y_score)
            ap_PU = average_precision_score(Y_test, y_score)
            acc_PU = accuracy_score(Y_test, y_pred)
            precision_PU = precision_score(Y_test, y_pred)
            recall_PU = recall_score(Y_test, y_pred)
            mcc = matthews_corrcoef(Y_test, y_pred)
            f1_PU = f1_score(Y_test, y_pred)
            H = caculate_avg(tree_H)
            print(f"tree_height_avg:{H}")
            Tree_H.append(H)
            print(f"AUC_PU:{auc_pu}, AP_PU:{ap_PU}, ACC_PU:{acc_PU}, precision_PU:{precision_PU}, recall_PU:{recall_PU}, mcc:{mcc}, f1_PU:{f1_PU}")
            Results_PU.append(auc_pu)
            N_ap_PU.append(ap_PU)
            N_acc_PU.append(acc_PU)
            N_precision_PU.append(precision_PU)
            N_recall_PU.append(recall_PU)
            N_mcc_PU.append(mcc)
            N_f1_PU.append(f1_PU)

        mean_Tree_H, mean_Results_GNAEMDA, mean_Results_PU, mean_N_ap_PU, mean_N_acc_PU, mean_N_f1_PU, mean_N_mcc_PU, mean_N_precision_PU, mean_N_recall_PU = caculate_avg(Tree_H),caculate_avg(Results_GNAEMDA),caculate_avg(Results_PU),caculate_avg(N_ap_PU),caculate_avg(N_acc_PU),caculate_avg(N_f1_PU),caculate_avg(N_mcc_PU),caculate_avg(N_precision_PU),caculate_avg(N_recall_PU)
        
        #print('Tree_Height_avg:', mean_Tree_H)

        print("--------------------")
        print('auc_PU:', mean_Results_PU)
        print("aupr_PU:",  mean_N_ap_PU)
        print("acc_PU:", mean_N_acc_PU)
        print("f1_PU:", mean_N_f1_PU)
        print("mcc_PU:", mean_N_mcc_PU)
        print("precision_PU:", mean_N_precision_PU)
        print("recall_PU:", mean_N_recall_PU)
        print("--------------------")
        M_auc_GAAE.append(mean_Results_GNAEMDA), M_Tree_H.append(mean_Tree_H)
        M_auc_PU.append(mean_Results_PU), M_aupr_PU.append(mean_N_ap_PU), M_acc_PU.append(mean_N_acc_PU), M_f1_PU.append(mean_N_f1_PU), M_mcc_PU.append(mean_N_mcc_PU), M_precision_PU.append(mean_N_precision_PU), M_recall_PU.append(mean_N_recall_PU)


    #print("M_auc_GAAE:", caculate_avg(M_auc_GAAE))
    print("M_Tree_H:", caculate_avg(M_Tree_H))
    print("--------------------")
    print("M_auc_PU:", caculate_avg(M_auc_PU))
    print("M_aupr_PU:", caculate_avg(M_aupr_PU))
    print("M_acc_PU:", caculate_avg(M_acc_PU))
    print("M_f1_PU:", caculate_avg(M_f1_PU))
    print("M_mcc_PU:", caculate_avg(M_mcc_PU))
    print("M_precision_PU:", caculate_avg(M_precision_PU))
    print("M_recall_PU:", caculate_avg(M_recall_PU))

main()
