import random

def caculate_avg(Results):
    sum = 0
    for i in Results:
        sum = sum + i
    return sum/len(Results)


def rondom_split(k, arr_data):
    arr = range(len(arr_data))
    every_len = int(len(arr) / k)
    arr_flag = []
    random_num = []
    index = 0
    for i in range(len(arr)):
        arr_flag.append(True)
        random_num.append(index)
        index += 1

    random.shuffle(random_num)

    result_arr = []
    every_arr = []
    index = 0
    for i in range(0, len(arr) - 1, every_len):
        index += 1
        for j in range(every_len):
            every_arr.append(arr[random_num[i]])
            i += 1
        result_arr.append(every_arr)
        every_arr = []
        if index >= k:
            break

    for i in range(len(random_num) - len(result_arr) * every_len):
        result_arr[i].append(arr[random_num[len(arr) - 1 - i]])
    all = []
    for index in result_arr:
        list1 = []
        for i in index:
            list1.append(arr_data[i])
        all.append(list1)
    return all

def cross_valid_experiment(adj, k, num_drug, num_microbe):
    pos_all = []
    neg_all = []
    pos_all_bip = []
    neg_all_bip = []
    for i in range(num_drug):
        for j in range(num_drug, num_drug + num_microbe):
            if adj[i][j] == 1:
                pos_all.append([i, j])
                pos_all_bip.append([i, j - num_drug])
            else:
                neg_all.append([i, j])
                if j > num_drug:
                    neg_all_bip.append([i, (j - num_drug)])
    pos_list_all = rondom_split(k, pos_all)
    pos_test = []
    neg_test = []
    for item in pos_list_all:
        pos_test.append(item)
        neg_test.append(random.sample(list(neg_all), len(item)))

    pos_list_all_Bip = rondom_split(k, pos_all_bip)
    pos_test_Bip = []
    neg_test_Bip = []
    for item in pos_list_all_Bip:
        pos_test_Bip.append(item)
        neg_test_Bip.append(random.sample(list(neg_all_bip), len(item)))

    return pos_test, neg_test,pos_test_Bip,neg_test_Bip,pos_all,neg_all


def process_emb(embs, adj, indexes,type='Hadamard'):
    label = []
    emb_reb = []
    for index in indexes:
        label.append(adj[index[0]][index[1]])
        if type=='Concatenate':
            emb_reb.append(np.concatenate((embs[index[0]].detach().cpu().numpy(), embs[index[1]].detach().cpu().numpy())))  # 拼接
        elif type== 'Average':
            emb_reb.append(embs[index[0]].detach().cpu().numpy() + embs[index[1]].detach().cpu().numpy())  # 逐位求和
        elif type=='L1-norm':
            emb_reb.append(np.fabs(embs[index[0]].detach().cpu().numpy() - embs[index[1]].detach().cpu().numpy()))  # L1范数
        elif type=='Hadamard':
            emb_reb.append(embs[index[0]].detach().cpu().numpy() * embs[index[1]].detach().cpu().numpy())  # Hadamard

    return emb_reb, label

