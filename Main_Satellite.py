# _*_ coding:   utf-8 _*_
# @Time     :    8:32 PM
import imageio
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import SLIC
import ASPPGCN
import cv2
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

samples_type = ['ratio', 'same_num'][0]


torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
F_Score_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []

Seed_List = [0, 1, 2, 3, 4]  # 随机种子点


data = cv2.imread('./Vaihigen/Results/images_vai1/top_mosaic_09cm_area1_3_0.tif')
gt = cv2.imread('./Vaihigen/Results/mask_vai1/top_mosaic_09cm_area1_3_0.tif', cv2.IMREAD_GRAYSCALE)
gt = gt + 1
mask = np.unique(gt)



train_ratio = 0.05  # 训练集比例。注意，训练集为按照‘每类’随机选取
val_ratio = 0.1  
class_count = 5  
learning_rate = 5e-4 
max_epoch = 600 
dataset_name = "Vaihingen_"
superpixel_scale=100


superpixel_scale = Scale  #########################
train_samples_per_class = 0.1  # 当定义为每类样本个数时,则该参数更改为训练样本数
val_samples = class_count
cmap = cm.get_cmap('jet', class_count + 1)
plt.set_cmap(cmap)
m, n, d = data.shape  


orig_data = data
height, width, bands = data.shape
data = np.reshape(data, [height * width, bands])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [height, width, bands])



def annotation2color(numlabel, name: str):
    # numlabel = np.array(label)
    color = np.zeros((*(256, 256), 3), dtype=np.uint8)
    color[numlabel == 1] = [0, 0, 255]
    color[numlabel == 2] = [0, 255, 0]
    color[numlabel == 3] = [0, 255, 255]
    color[numlabel == 4] = [255, 255, 0]
    color[numlabel == 5] = [255, 255, 255]

    imageio.imwrite(name + '_RGB.png', color, format='png')
    pass


def Draw_Classification_Map(label, name, scale = 4.0, dpi = 660):
 
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    a = annotation2color(numlabel, name)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    pass


def GT_To_One_Hot(gt, class_count):
    '''
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            # temp[int(gt[i, j])] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


for curr_seed in Seed_List:
    random.seed(curr_seed)
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)
        test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    if samples_type == 'same_num':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = train_samples_per_class
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_samples)  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)

        test_data_index = test_data_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
    test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
    val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
    test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
    val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

    # 训练集
    train_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    train_samples_gt = np.reshape(train_samples_gt, [m * n])
    for i in range(m * n):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

    # 测试集
    test_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    test_samples_gt = np.reshape(test_samples_gt, [m * n])
    for i in range(m * n):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

    # 验证集
    val_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    val_samples_gt = np.reshape(val_samples_gt, [m * n])
    for i in range(m * n):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

    ls = SLIC.LDA_SLIC(data, np.reshape(train_samples_gt, [height, width]), class_count - 1)
    tic0 = time.time()
    Q, S, A, Seg = ls.simple_superpixel_no_LDA(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    # np.save(dataset_name+'Seg',Seg)
    print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)

    # 转到GPU
    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
    # 转到GPU
    train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
    # 转到GPU
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)


    if dataset_name == "Vaihingen_":
        net = ASPPGCN.ASPPGCN(height, width, bands, class_count, Q, A, model='smoothed')
    else:
        net = ASPPGCN.ASPPGCN(height, width, bands, class_count, Q, A)

    print("parameters", net.parameters(), len(list(net.parameters())))
    net.to(device)


    def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
        real_labels = reallabel_onehot
        we = -torch.mul(real_labels, torch.log(predict))
        we = torch.mul(we, reallabel_mask)
        pool_cross_entropy = torch.sum(we)
        return pool_cross_entropy


    zeros = torch.zeros([m * n]).to(device).float()


    def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                             printFlag=True):
        if False == require_AA_KPP:
            with torch.no_grad():
                available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                available_label_count = available_label_idx.sum()  # 有效标签的个数
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction.cpu() / available_label_count

                return OA
        else:
            with torch.no_grad():
                # 计算OA
                available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                available_label_count = available_label_idx.sum()  # 有效标签的个数
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                    available_label_idx, zeros).sum()
                OA = correct_prediction.cpu() / available_label_count
                OA = OA.cpu().numpy()

                # 计算AA
                zero_vector = np.zeros([class_count])
                output_data = network_output.cpu().numpy()
                train_samples_gt = train_samples_gt.cpu().numpy()
                train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                output_data = np.reshape(output_data, [m * n, class_count])
                idx = np.argmax(output_data, axis=-1)
                for z in range(output_data.shape[0]):
                    if ~(zero_vector == output_data[z]).all():
                        idx[z] += 1
                # idx = idx + train_samples_gt
                count_perclass = np.zeros([class_count])
                correct_perclass = np.zeros([class_count])
                for x in range(len(train_samples_gt)):
                    if train_samples_gt[x] != 0:
                        count_perclass[int(train_samples_gt[x] - 1)] += 1
                        if train_samples_gt[x] == idx[x]:
                            correct_perclass[int(train_samples_gt[x] - 1)] += 1
                test_AC_list = correct_perclass / count_perclass
                test_AA = np.average(test_AC_list)


                # 计算F1-score

                test_pre_label_list = []
                test_real_label_list = []
                output_data = np.reshape(output_data, [m * n, class_count])
                idx = np.argmax(output_data, axis=-1)
                idx = np.reshape(idx, [m, n])
                for ii in range(m):
                    for jj in range(n):
                        if Test_GT[ii][jj] != 0:
                            test_pre_label_list.append(idx[ii][jj] + 1)
                            test_real_label_list.append(Test_GT[ii][jj])
                test_pre_label_list = np.array(test_pre_label_list)
                test_real_label_list = np.array(test_real_label_list)
                F_score = metrics.f1_score(test_pre_label_list.astype(np.int16), test_real_label_list.astype(np.int16), average='macro')
                test_F = F_score

                # 计算KPP
                test_pre_label_list = []
                test_real_label_list = []
                output_data = np.reshape(output_data, [m * n, class_count])
                idx = np.argmax(output_data, axis=-1)
                idx = np.reshape(idx, [m, n])
                for ii in range(m):
                    for jj in range(n):
                        if Test_GT[ii][jj] != 0:
                            test_pre_label_list.append(idx[ii][jj] + 1)
                            test_real_label_list.append(Test_GT[ii][jj])
                test_pre_label_list = np.array(test_pre_label_list)
                test_real_label_list = np.array(test_real_label_list)
                kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                  test_real_label_list.astype(np.int16))
                test_kpp = kappa

                # 输出
                if printFlag:
                    print("test OA=", OA, "AA=", test_AA, "F_scores=", test_F, 'kpp=', test_kpp)
                    print('acc per class:')
                    print(test_AC_list)

                OA_ALL.append(OA)
                AA_ALL.append(test_AA)
                F_Score_ALL.append(test_F)
                KPP_ALL.append(test_kpp)
                AVG_ALL.append(test_AC_list)

                # 保存数据信息
                f = open('./Vai11_results/' + dataset_name + '_results.txt', 'a+')
                str_results = '\n======================' \
                              + " learning rate=" + str(learning_rate) \
                              + " epochs=" + str(max_epoch) \
                              + " train ratio=" + str(train_ratio) \
                              + " val ratio=" + str(val_ratio) \
                              + " ======================" \
                              + "\nOA=" + str(OA) \
                              + "\nAA=" + str(test_AA) \
                              + "\nF_scores=" + str(test_F) \
                              + '\nkpp=' + str(test_kpp) \
                              + '\nacc per class:' + str(test_AC_list) + "\n"
                # + '\ntrain time:' + str(time_train_end - time_train_start) \
                # + '\ntest time:' + str(time_test_end - time_test_start) \
                f.write(str_results)
                f.close()
                return OA


    # 训练
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # ,weight_decay=0.0001
    best_loss = 99999
    net.train()
    tic1 = time.perf_counter()
    for i in range(max_epoch + 1):
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(net_input)
        loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update
        if i % 10 == 0:
            with torch.no_grad():
                net.eval()
                output = net(net_input)
                trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                print(
                    "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                     valloss, valOA))

                if valloss < best_loss:
                    best_loss = valloss
                    torch.save(net.state_dict(), "./model/best_model_Vai11.pt")
                    print('save model...')
            torch.cuda.empty_cache()
            net.train()
    toc1 = time.perf_counter()
    print("\n\n====================training done. starting evaluation...========================\n")
    training_time = toc1 - tic1 + LDA_SLIC_Time  # 分割耗时需要算进去
    Train_Time_ALL.append(training_time)

    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load("./model/best_model_Vai11.pt"))
        net.eval()
        tic2 = time.perf_counter()
        output = net(net_input)
        toc2 = time.perf_counter()
        testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
        testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                      printFlag=False)
        print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
        # 计算
        classification_map = torch.argmax(output, 1).reshape([height, width]).cpu() + 1
        Draw_Classification_Map(classification_map, "./Vai11_results/" + dataset_name + str(testOA))
        testing_time = toc2 - tic2 + LDA_SLIC_Time  # 分割耗时需要算进去
        Test_Time_ALL.append(testing_time)


    torch.cuda.empty_cache()
    del net

OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
F_score_ALL = np.array(F_score_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

print("\ntrain_ratio={}".format(curr_train_ratio),
      "\n==============================================================================")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('F_Score=', np.mean(F_Score_ALL), '+-', np.std(F_Score_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
print("Average training time:{}".format(np.mean(Train_Time_ALL)))
print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

# 保存数据信息
f = open('./Vai11_results/' + dataset_name + '_results.txt', 'a+')
str_results = '\n\n************************************************' \
              + "\ntrain_ratio={}".format(curr_train_ratio) \
              + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
              + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
              + '\nAA=' + str(np.mean(F_Score_ALL)) + '+-' + str(np.std(F_Score_ALL)) \
              + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
              + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
              + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
              + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
f.write(str_results)
f.close()









