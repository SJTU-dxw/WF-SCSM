import argparse
import glob
import numpy as np
import multiprocessing as mp
import tqdm
import pandas as pd
import os
import shutil
import pickle
import json
import random
import torch
import torch.utils.data as Data
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


TAM_LENGTH = 1000
CUTOFF_TIME = 80
ROUND = 1
SET_SIZE = 30
alpha_upload = 0.16
alpha_download = 0.16
U_upload = 45
U_download = 45
B = 20
eps = 1e-6
TIME_SLOT = CUTOFF_TIME / TAM_LENGTH


def parse_arguments():
    parser = argparse.ArgumentParser(description='It simulates adaptive padding on a set of web traffic traces.')

    parser.add_argument('--traces_path',
                        metavar='<traces path>',
                        default="../../Swallow/Traces/D1-Undefence",
                        help='Path to the directory with the traffic traces to be simulated.')

    # Parse arguments
    args = parser.parse_args()
    return args


def parallel(para_list, n_jobs=10):
    pool = mp.Pool(n_jobs)
    data_dict = tqdm.tqdm(pool.imap(extract_feature, para_list), total=len(para_list))
    pool.close()
    return data_dict


def packets_per_slot(times, sizes):
    feature = [[0 for _ in range(TAM_LENGTH)], [0 for _ in range(TAM_LENGTH)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= CUTOFF_TIME:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (TAM_LENGTH - 1) / CUTOFF_TIME)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= CUTOFF_TIME:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (TAM_LENGTH - 1) / CUTOFF_TIME)
                feature[1][idx] += 1

    return feature


def extract_feature(f):
    file_name = f.split('/')[-1]

    with open(f, 'r') as f:
        tcp_dump = f.readlines()

    seq = pd.Series(tcp_dump[:]).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    times = np.array(seq.iloc[:, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")
    length_seq = np.sign(length_seq)
    feature = packets_per_slot(times, length_seq)
    if '-' in file_name:
        label = file_name.split('-')
        label = int(label[0])
    else:
        label = -1

    return feature, label


def extract_TAM(args):
    # parser config and arguments
    para_list = glob.glob(os.path.join(args.traces_path, "*"))

    data_dict = {'dataset': [], 'label': []}

    raw_data_dict = parallel(para_list, n_jobs=15)
    features, label = zip(*raw_data_dict)

    features = np.array(features)
    labels = np.array(label)

    # features = features[labels != -1]
    # labels = labels[labels != -1]
    # if -1 in labels:
    #     labels[labels == -1] = np.max(labels) + 1

    data_dict['dataset'], data_dict['label'] = features, labels

    output_dir = args.traces_path.rstrip("/") + "_TAM"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir + "/data.npy", data_dict)
    print('save to %s' % (output_dir + "/data.npy"))


def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    print(train_X.shape, train_y.shape)

    return train_X, train_y


def getDict(dataset, website_indices):
    website_dict = {}
    for i in range(len(dataset)):
        if website_indices[i] not in website_dict.keys():
            website_dict[website_indices[i]] = dataset[i: i + 1]
        else:
            website_dict[website_indices[i]] = np.append(website_dict[website_indices[i]], dataset[i: i + 1], axis=0)

    return website_dict


def get_matrix(dataset_path):
    train_X_ori, train_y_ori = load_data(dataset_path)
    print(np.bincount(train_y_ori))

    website_dict = getDict(train_X_ori, train_y_ori)

    super_matrices_websites = np.zeros((len(np.unique(train_y_ori)), 2, TAM_LENGTH))

    # build the super-matrix for each website using the corresponding traces
    for website in sorted(website_dict.keys()):
        super_matrix_website = np.max(website_dict[website], axis=0)
        super_matrices_websites[website] = super_matrix_website

    return super_matrices_websites


def update_matrix(matrix_1, matrix_2):
    updated_matrix = np.maximum(matrix_1, matrix_2)

    return updated_matrix


def website_clustering(partition, super_matrices, k=5):
    if len(partition) == 0:
        return [], []

    partition_set = []
    centers = np.empty((0, 2, TAM_LENGTH), float)
    partition = np.sort(partition)
    tar_label = np.random.choice(partition, 1)[0]
    visited = [0] * len(super_matrices)
    visited[tar_label] = 1
    partition_set.append([tar_label])
    centers = np.append(centers, super_matrices[tar_label:tar_label + 1], axis=0)
    node = 0
    cnt = 1

    while cnt < len(partition):

        if len(partition_set[node]) == k:
            if len(partition) - cnt < k:
                break
            node += 1
            max_dis = -1
            max_idx = -1
            for i in partition:
                if visited[i] == 1:
                    continue
                for ct in centers:
                    dis = np.linalg.norm(super_matrices[i] - ct)
                    if dis > max_dis:
                        max_dis = dis
                        max_idx = i

            if max_idx == -1:
                break
            else:
                partition_set.append([max_idx])
                centers = np.append(centers, super_matrices[max_idx][np.newaxis, :], axis=0)
                visited[max_idx] = 1
                cnt += 1

        min_dis = 1e9
        min_idx = -1

        for i in partition:
            if visited[i] == 1:
                continue
            if min_dis > np.linalg.norm(super_matrices[i] - centers[node]):
                min_dis = np.linalg.norm(super_matrices[i] - centers[node])
                min_idx = i

        if min_idx == -1:
            break
        else:
            visited[min_idx] = 1
            partition_set[node].append(min_idx)
            centers[node] = update_matrix(centers[node], super_matrices[min_idx])
            cnt += 1

    for i in partition:
        if visited[i] == 0:
            min_dis = 1e9
            min_idx = -1
            for idx, ct in enumerate(centers):
                dis = np.linalg.norm(super_matrices[i] - ct)
                if dis < min_dis:
                    min_dis = dis
                    min_idx = idx

            partition_set[min_idx].append(i)
            centers[min_idx] = update_matrix(centers[min_idx], super_matrices[i])

    return partition_set, centers


def drawLine(x, y, colors, label):
    plt.plot(x, y, color=colors, label=label)


def load_data_npy(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    return train_X, train_y


def getArray(line, row):
    trace_cum = []

    for i in range(line):
        sub_arr = []
        for j in range(row):
            sub_arr.append(0)
        trace_cum.append(sub_arr)
    return trace_cum


def draw_plot(super_matrices, anonymity_set, tam_len, output_path):
    fig = plt.figure(figsize=(8, 5))
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'orange', 'purple', 'brown', 'gray']
    for i in range(len(anonymity_set)):
        drawLine(range(0, tam_len), super_matrices[anonymity_set[i]][0, :tam_len], colors[i % 10],
                 label=anonymity_set[i])
        drawLine(range(0, tam_len), -super_matrices[anonymity_set[i]][1, :tam_len], colors[i % 10],
                 label=anonymity_set[i])
    plt.ylim(-1000, 500)
    fig.savefig(output_path)
    plt.close(fig)


def get_PMF(dataset_dir, data_dict, set_num, tam_len):
    feature, label = load_data_npy(dataset_dir)
    arr = [[] for i in range(set_num)]
    print(len(label))
    for i in range(len(label)):
        index = int(label[i])
        for j in range(len(data_dict[index])):
            arr[data_dict[index][j]].append(i)

    trace_cum_upload = getArray(set_num, tam_len)
    trace_cum_download = getArray(set_num, tam_len)
    for i in range(len(arr)):
        trace_sum_res_upload = [0] * tam_len
        trace_sum_res_download = [0] * tam_len
        print(len(arr[i]))
        for j in range(len(arr[i])):
            trace = feature[arr[i][j]]

            trace_upload = trace[0]
            trace_download = trace[1]
            new_trace_upload = [1 if i != 0 else 0 for i in trace_upload]

            new_trace_download = [1 if i != 0 else 0 for i in trace_download]
            trace_sum_res_upload = [x + y for x, y in zip(new_trace_upload, trace_sum_res_upload)]
            trace_sum_res_download = [x + y for x, y in zip(new_trace_download, trace_sum_res_download)]

        trace_cum_upload[i] = trace_sum_res_upload
        trace_cum_download[i] = trace_sum_res_download

    trace_cum_res_upload = getArray(set_num, tam_len)
    trace_cum_res_download = getArray(set_num, tam_len)
    for i in range(len(trace_cum_upload)):
        normalized_arr_upload = trace_cum_upload[i] / np.sum(trace_cum_upload[i])
        trace_cum_res_upload[i] = normalized_arr_upload
        normalized_arr_download = trace_cum_download[i] / np.sum(trace_cum_download[i])
        trace_cum_res_download[i] = normalized_arr_download

    return trace_cum_res_upload, trace_cum_res_download


def cluster(args):
    # list of generated anonymity sets
    total_sets = []
    # list of generated super-matrices, the super-matrix of total_sets[i] is super_matrices[i]
    super_matrices = []
    # mapping of website to anonymity set index
    website_to_set = {}

    # 每个网站各个位置的最大值
    super_matrices_websites = get_matrix(args.traces_path + "_TAM/data.npy")
    website_indices = np.array([i for i in range(len(super_matrices_websites))])
    np.random.shuffle(website_indices)

    partition_1 = website_indices
    partition_2 = []
    for i in range(ROUND):
        anonymity_sets_fir, super_matrices_fir = website_clustering(partition_1, super_matrices_websites, SET_SIZE)
        anonymity_sets_sec, super_matrices_sec = website_clustering(partition_2, super_matrices_websites, SET_SIZE)

        partition_1 = []
        partition_2 = []

        for anonymity_set, super_matrix in zip(anonymity_sets_fir, super_matrices_fir):
            super_matrix = np.where(super_matrix == 0, 1, super_matrix)
            for website in anonymity_set:
                if len(partition_1) < len(partition_2):
                    partition_1.append(website)
                else:
                    partition_2.append(website)
            if anonymity_set in total_sets:
                continue
            total_sets.append(anonymity_set)
            super_matrices.append(super_matrix)

        for anonymity_set, super_matrix in zip(anonymity_sets_sec, super_matrices_sec):
            super_matrix = np.where(super_matrix == 0, 1, super_matrix)
            for website in anonymity_set:
                if len(partition_1) < len(partition_2):
                    partition_1.append(website)
                else:
                    partition_2.append(website)
            if anonymity_set in total_sets:
                continue
            total_sets.append(anonymity_set)
            super_matrices.append(super_matrix)

        partition_1 = np.array(partition_1)
        partition_2 = np.array(partition_2)

    print(total_sets)

    for idx, anonymity_set in enumerate(total_sets):
        for website in anonymity_set:
            if website in website_to_set.keys():
                website_to_set[website].append(idx)
            else:
                website_to_set[website] = [idx]

    print(website_to_set)

    # you can visualize the super-matrix of websites in each anonymity set
    plot_dir = os.path.join(args.traces_path.rstrip("/") + "_TAM", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for set_idx, per_set in enumerate(total_sets):
        draw_plot(super_matrices_websites, per_set, TAM_LENGTH,
                  os.path.join(plot_dir, "anonymity_set_{:03d}.png".format(set_idx)))

    # PMF estimation
    PMF_upload, PMF_download = get_PMF(args.traces_path + "_TAM/data.npy", website_to_set, len(total_sets),
                                       TAM_LENGTH)

    pickle.dump(PMF_upload,
                open(args.traces_path + '_TAM/PMF_upload_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                                          str(ROUND)) + '.pkl', 'wb'))

    PMF_upload = list(map(lambda arr: arr.tolist(), PMF_upload))
    with open(args.traces_path + '_TAM/PMF_upload_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                str(ROUND)) + '.json', 'w') as f:
        json.dump(PMF_upload, f)

    pickle.dump(PMF_download,
                open(args.traces_path + '_TAM/PMF_download_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                                            str(ROUND)) + '.pkl', 'wb'))
    PMF_download = list(map(lambda arr: arr.tolist(), PMF_download))
    with open(args.traces_path + '_TAM/PMF_download_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                  str(ROUND)) + '.json', 'w') as f:
        json.dump(PMF_download, f)

    pickle.dump(total_sets,
                open(args.traces_path + '_TAM/total_set_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                                         str(ROUND)) + '.pkl', 'wb'))
    np.save(args.traces_path + '_TAM/super_matrices_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                                     str(ROUND)) + '.npy', super_matrices)
    np.save(args.traces_path + '_TAM/website_to_set_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE),
                                                                     str(ROUND)) + '.npy', website_to_set)

    with open(args.traces_path + '_TAM/website_to_set.json', 'w') as f:
        f.write(str(website_to_set))


def load_data_tensor(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']

    train_X = torch.from_numpy(train_X).type(torch.FloatTensor)
    train_X = train_X.view(train_X.size(0), 1, 2, -1)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    print(train_X.shape, train_y.shape)

    return train_X, train_y


def get_from_set(train_X, train_Y, st, out_dim=1000, use_mapping=True):
    sorted_st = sorted(st)
    new_train_X = torch.empty(0, 1, 2, out_dim)
    new_train_Y = torch.empty(0, dtype=torch.long)
    label_mapping = []
    for idx, label in enumerate(sorted_st):
        label_mapping.append(label)
        now_X = train_X[train_Y == label]
        length = now_X.shape[0]
        new_train_X = torch.cat((new_train_X, train_X[train_Y == label]), 0)
        if use_mapping:
            new_train_Y = torch.cat((new_train_Y, torch.tensor([idx] * length)), 0)
        else:
            new_train_Y = torch.cat((new_train_Y, torch.tensor([label] * length)), 0)

    label_mapping = torch.LongTensor(label_mapping).cuda()

    return new_train_X, new_train_Y, label_mapping


def refine(args):
    batch_size = 32 * 8
    Epoch = 100
    lr = 0.0001
    seed = 1
    anonymity_dir = args.traces_path + '_TAM/'
    anonymity_suffix = '_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE), str(ROUND))

    random.seed(seed)

    total_set = pickle.load(open(anonymity_dir + 'total_set' + anonymity_suffix + '.pkl', 'rb'))

    super_matrices = np.load(anonymity_dir + 'super_matrices' + anonymity_suffix + '.npy', allow_pickle=True)
    print(super_matrices)
    train_X, train_y = load_data_tensor(args.traces_path + "_TAM/data.npy")
    test_X, test_y = train_X, train_y

    shrunk_super_matrices = []

    for idx, st in enumerate(total_set):
        ct = torch.from_numpy(super_matrices[idx][np.newaxis, np.newaxis, :]).type(torch.FloatTensor).cuda()
        new_train_X, new_train_y, train_label_mapping = get_from_set(train_X, train_y, st, out_dim=TAM_LENGTH)
        new_test_X, new_test_y, test_label_mapping = get_from_set(test_X, test_y, st, out_dim=TAM_LENGTH,
                                                                  use_mapping=False)

        train_dataset = Data.TensorDataset(new_train_X, new_train_y)
        test_dataset = Data.TensorDataset(new_test_X, new_test_y)

        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size * 5, shuffle=True, num_workers=0)

        weight = torch.randn(2, TAM_LENGTH).cuda()
        bias = torch.zeros(2, TAM_LENGTH).cuda()

        adv_weight = weight.clone().detach()
        adv_bias = bias.clone().detach()

        adv_weight.requires_grad = True
        adv_bias.requires_grad = True

        th = torch.tensor([[10.], [10.]]).cuda()
        th = torch.where(ct > th, th, ct)

        optimizer = torch.optim.Adam([adv_weight, adv_bias], lr=lr)

        for epoch in range(Epoch):
            for batch_idx, (tr_x, _) in enumerate(train_loader):
                tr_x = tr_x.cuda()
                optimizer.zero_grad()

                adv_weight.requires_grad = True
                adv_bias.requires_grad = True

                trans_weight = torch.sigmoid(adv_weight)
                trans_bias = th * torch.sigmoid(adv_bias)

                pruned = torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0)

                loss = - 0.5 * torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, max=0)) \
                       + torch.mean(torch.clamp(pruned * torch.sign(tr_x) - tr_x, min=10.))

                loss.backward()
                optimizer.step()

        trans_weight = torch.sigmoid(adv_weight)
        trans_bias = th * torch.sigmoid(adv_bias)

        pruned_ct = torch.ceil(torch.clamp(torch.clamp(ct * trans_weight, min=th) - trans_bias, min=0))
        shrunk_super_matrices.append(pruned_ct.detach().cpu().numpy().astype(np.int16))

    np.save(args.traces_path + '_TAM/shrunk_super_matrices' + anonymity_suffix + '.npy', shrunk_super_matrices,
            allow_pickle=True)

    shrunk_super_matrices = np.squeeze(shrunk_super_matrices).astype(int)
    with open(args.traces_path + '_TAM/shrunk_super_matrices' + anonymity_suffix + '.json', 'w') as f:
        json.dump(shrunk_super_matrices.tolist(), f)


def sample(trace_prob, threshold, tam_len):
    slot_idx = list(range(0, tam_len))
    random.shuffle(slot_idx)
    sampled_slots = []
    cum_prob = 0
    for i in range(tam_len):
        if cum_prob >= threshold:
            return sampled_slots
        cum_prob = cum_prob + trace_prob[slot_idx[i]]
        sampled_slots.append(slot_idx[i])
    return sampled_slots


def parallel_2(para_list, n_jobs=15):
    pool = mp.Pool(n_jobs)
    overheads = tqdm.tqdm(pool.imap(generate_defense, para_list), total=len(para_list))
    pool.close()

    return overheads


def generate_defense(para):
    cur_shrunk_super_matrix, now_dir, now_file_path, sampled_slots_upload, sampled_slots_download, is_dump = para
    now_file_name = now_file_path.split('/')[-1]

    with open(now_file_path, 'r') as f:
        tcp_dump = f.readlines()

    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype(
        "float")
    times = np.array(seq.iloc[:, 0]) - np.array(seq.iloc[0, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")
    length_seq = np.sign(length_seq)

    packets = np.empty((0, 2), dtype=np.float32)

    now_timestamp = TIME_SLOT
    # current slot index
    now_slot_idx = 0

    # the number of buffered packets
    cum_upload = 0.
    cum_download = 0.

    # total real packets
    total_pkt = 0.
    # total upload packets
    total_pkt_upload = 0.
    # total download packets
    total_pkt_download = 0.

    # the last time slot which has real packets
    final_slot = 0.

    # the current packet sequences idx
    now_sequence_idx = 0

    # threshold for delayed packets
    u_upload = random.randint(0, U_upload - 1)
    u_download = random.randint(0, U_download - 1)

    # a flag which indicates if there do not have any buffered packets
    flag_end_upload = False
    flag_end_download = False

    # a flag which indicates if the first upload/download packet has been sent
    flag_first_upload = False
    flag_first_download = False

    while now_timestamp <= CUTOFF_TIME:

        # the sending budget in next u slots
        budget_upload = np.sum(cur_shrunk_super_matrix[0, now_slot_idx: now_slot_idx + u_upload])
        budget_download = np.sum(cur_shrunk_super_matrix[1, now_slot_idx: now_slot_idx + u_download])

        sm_upload = cur_shrunk_super_matrix[0, now_slot_idx]
        sm_download = cur_shrunk_super_matrix[1, now_slot_idx]

        target_upload = sm_upload
        target_download = sm_download

        # super-matrix sampling
        if now_slot_idx not in sampled_slots_upload:
            target_upload = 0.
        if now_slot_idx not in sampled_slots_download:
            target_download = 0.

        # get the number of real packets sent at current slot
        while now_sequence_idx < len(times) and times[now_sequence_idx] < now_timestamp:
            total_pkt += 1
            if length_seq[now_sequence_idx] > 0:
                cum_upload += 1
                total_pkt_upload += 1
            elif length_seq[now_sequence_idx] < 0:
                cum_download += 1
                total_pkt_download += 1
            now_sequence_idx += 1

        if cum_upload != 0.:
            final_slot = now_slot_idx
            flag_end_upload = False
        if cum_download != 0.:
            final_slot = now_slot_idx
            flag_end_download = False

        # Regulate upload packets
        send_upload = target_upload
        # if no buffered real packets, we will check if the current time slot idx is multiple of B
        if cum_upload == 0:
            # send packets until real packets arrive
            if flag_end_upload:
                send_upload = 0
            elif now_slot_idx != 0 and now_slot_idx % B == 0:
                flag_end_upload = True

        if total_pkt_upload >= 1 and flag_first_upload:
            # early sending
            if cum_upload != 0 and cum_upload >= budget_upload:
                send_upload = max(min(10, cum_upload), sm_upload)
        else:
            if cum_upload != 0:
                send_upload = max(cum_upload, send_upload)
            else:
                send_upload = 0

        # Regulate download packets
        send_download = target_download
        if cum_download == 0:
            if flag_end_download:
                send_download = 0
            elif now_slot_idx != 0 and now_slot_idx % B == 0:
                flag_end_download = True

        if total_pkt_download >= 1 and flag_first_download:
            if cum_download != 0 and cum_download >= budget_download:
                send_download = max(min(30, cum_download), sm_download)
        else:
            if cum_download != 0:
                send_download = max(cum_download, send_download)
            else:
                send_download = 0

        if total_pkt_upload >= 1:
            flag_first_upload = True
        if total_pkt_download >= 1:
            flag_first_download = True

        cum_upload = max(0., cum_upload - send_upload)
        cum_download = max(0., cum_download - send_download)

        # sample the timestamps for packets sent in current slot
        upload_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.03, int(send_upload)), a_min=now_timestamp,
                                    a_max=now_timestamp + TAM_LENGTH - eps)
        download_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.03, int(send_download)),
                                      a_min=now_timestamp, a_max=now_timestamp + TAM_LENGTH - eps)

        for j in range(len(upload_timestamps)):
            packets = np.append(packets, np.array([[upload_timestamps[j], 1]]), axis=0)
        for j in range(len(download_timestamps)):
            packets = np.append(packets, np.array([[download_timestamps[j], -1]]), axis=0)

        now_slot_idx += 1
        now_timestamp = TIME_SLOT * (now_slot_idx + 1)

    if cum_upload > 0.:
        final_slot = now_slot_idx
        upload_timestamps = np.clip(now_timestamp + np.random.rayleigh(0.1, int(cum_upload)), a_min=now_timestamp,
                                    a_max=now_timestamp + 5.)
        for p in range(len(upload_timestamps)):
            packets = np.append(packets, np.array([[upload_timestamps[p], 1]]), axis=0)

    if cum_download > 0.:
        final_slot = now_slot_idx
        download_timestamps = np.clip(np.random.rayleigh(0.1, int(cum_download)), a_min=now_timestamp,
                                      a_max=now_timestamp + 5.)
        for p in range(len(download_timestamps)):
            packets = np.append(packets, np.array([[download_timestamps[p], -1]]), axis=0)

    packets_idx = np.argsort(packets[:, 0])
    packets = packets[packets_idx]
    packets[:, 0] = packets[:, 0] - packets[0, 0]

    if is_dump:
        dump(packets, now_dir, now_file_name)

    final_end = max((final_slot + 1) * TIME_SLOT, np.minimum(CUTOFF_TIME, times[-1]))

    return len(packets), total_pkt, final_end, np.minimum(CUTOFF_TIME, times[-1])


def dump(trace, output_path, file):
    with open(os.path.join(output_path, file), 'w') as fo:
        for i in range(len(trace)):
            fo.write("{}".format(trace[i][0]) + '\t' + "{}".format(int(trace[i][1]) * 512) + '\n')


def regularization(args):
    anonymity_dir = args.traces_path + '_TAM/'
    anonymity_suffix = '_{}_{}_{}'.format(str(TAM_LENGTH), str(SET_SIZE), str(ROUND))

    n_jobs = 10
    seed = 1

    random.seed(seed)

    website_to_set = np.load(anonymity_dir + 'website_to_set' + anonymity_suffix + '.npy', allow_pickle=True).item()
    shrunk_super_matrices = np.load(anonymity_dir + 'shrunk_super_matrices' + anonymity_suffix + '.npy',
                                    allow_pickle=True)
    PMF_upload = pickle.load(open(anonymity_dir + 'PMF_upload' + anonymity_suffix + '.pkl', 'rb'))
    PMF_download = pickle.load(open(anonymity_dir + 'PMF_download' + anonymity_suffix + '.pkl', 'rb'))

    process_data = []
    para_list = glob.glob(os.path.join(args.traces_path, "*"))
    for file in para_list:
        file_name = file.split("/")[-1].split(".")[0]
        i, j = file_name.split("-")
        i = int(i)
        j = int(j)

        website_idx = i

        anonymity_sets = website_to_set[website_idx]
        anonymity_set_idx = anonymity_sets[j % len(anonymity_sets)]

        shrunk_super_matrix = np.ceil(shrunk_super_matrices[anonymity_set_idx])
        np.set_printoptions(threshold=np.inf)
        sampled_slots_upload = sample(PMF_upload[anonymity_set_idx], alpha_upload, tam_len=TAM_LENGTH)
        sampled_slots_download = sample(PMF_download[anonymity_set_idx], alpha_download, tam_len=TAM_LENGTH)
        print(len(sampled_slots_upload), len(sampled_slots_download))

        output_dir = args.traces_path.rstrip("/") + "_Palette"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        process_data.append((shrunk_super_matrix[0, 0, :, :], output_dir, file, sampled_slots_upload,
                             sampled_slots_download, True))

    overheads = parallel_2(process_data, n_jobs=n_jobs)
    final_band, pre_band, final_time, pre_time = zip(*overheads)

    print(np.sum(np.array(final_band)) / np.sum(np.array(pre_band)) - 1,
          np.sum(np.array(final_time)) / np.sum(np.array(pre_time)) - 1)


if __name__ == "__main__":
    args = parse_arguments()
    print("Arguments: %s" % args)

    # 构建TAM数据集，注意，cell必须是k-n的形式，只能是closed-world，注意输出被转换为+-512（固定1000shots，80s）
    tam_data_path = args.traces_path.rstrip("/") + "_TAM/data.npy"
    if os.path.exists(tam_data_path):
        print("Reuse existing TAM data: %s" % tam_data_path)
    else:
        extract_TAM(args)

    # 对网站依据TAM进行聚类
    cluster(args)

    # 对每个类别训练，获取shrunk_super_matrices
    refine(args)

    # 产生防御traces
    regularization(args)

    # 删除中间目录
    shutil.rmtree(args.traces_path + '_TAM/')
