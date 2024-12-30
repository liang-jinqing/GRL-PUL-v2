import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tools import process_emb


def sel_neg_from_rn(rn_ij, num, unlabelled_ij):
    if len(rn_ij) >= num:
        neg_sel_idx = np.random.choice(len(rn_ij), num, replace=False)
        rn_ij_np = np.array(rn_ij)
        neg_sel = rn_ij_np[neg_sel_idx]
    else:
        unlabelled_ij_tuples = [tuple(inner_list) for inner_list in unlabelled_ij]
        rn_ij_tuples = [tuple(inner_list) for inner_list in rn_ij]
        possible_neg = np.array(list(set(unlabelled_ij_tuples) - set(rn_ij_tuples)))
        neg_idx = np.random.choice(len(possible_neg), num-len(rn_ij), replace=False)
        neg_sel_np = possible_neg[neg_idx]
        neg_sel = [[int(item) for item in sublist] for sublist in neg_sel_np]
        if len(rn_ij)!=0:
            rn_ij = [[int(item) for item in sublist] for sublist in rn_ij]
            neg_sel = neg_sel + rn_ij

    return neg_sel


def sel_neg_by_bagging(pos_ij, unlabelled_ij, feature, adj_np, iterate_time):

    num_pos = len(pos_ij)
    prob_mat = np.zeros_like(adj_np, dtype=float)
    cnt = np.zeros_like(adj_np)
    for t in range(iterate_time):
        print(f"t:{t}")
        unlabelled_train_ij_t_idx = np.random.choice(len(unlabelled_ij), num_pos, replace=False)
        unlabelled_ij = np.array(unlabelled_ij)
        unlabelled_train_ij_t = unlabelled_ij[unlabelled_train_ij_t_idx]
        rest_unlabelled_train_ij_t_idx = np.setdiff1d(np.arange(len(unlabelled_ij)), unlabelled_train_ij_t_idx)
        rest_unlabelled_train_ij_t = unlabelled_ij[rest_unlabelled_train_ij_t_idx]
        train_ij = np.vstack((pos_ij, unlabelled_train_ij_t))
        train_feat, train_label = process_emb(feature, adj_np, train_ij)
        rest_unlabelled_train_feat, rest_unlabelled_train_label = process_emb(feature, adj_np, rest_unlabelled_train_ij_t)
        regressor = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        regressor.fit(train_feat, train_label)
        rest_unlabelled_train_prob = regressor.predict_proba(rest_unlabelled_train_feat)[:, 1]
        prob_mat[tuple(list(rest_unlabelled_train_ij_t.T))] += rest_unlabelled_train_prob
        cnt[tuple(list(rest_unlabelled_train_ij_t.T))] += 1

    fnl_score = prob_mat / cnt
    non_nan_indexes = np.nonzero(~np.isnan(fnl_score))
    non_nan_values = fnl_score[non_nan_indexes]
    non_nan_ij = np.transpose(non_nan_indexes)
    pos_feat, _ = process_emb(feature, adj_np, pos_ij)
    pos_prob = regressor.predict_proba(pos_feat)[:, 1]
    pos_prob_thresh = np.sort(pos_prob)[int(len(pos_ij) * 0.02)]
    sorted_nums = sorted(enumerate(non_nan_values), key=lambda x: x[1])
    neg_nums = list(filter(lambda x: x[1] < pos_prob_thresh, sorted_nums))
    idx = [i[0] for i in neg_nums]
    rn_ij = non_nan_ij[idx]

    return sel_neg_from_rn(rn_ij, num_pos, unlabelled_ij)


