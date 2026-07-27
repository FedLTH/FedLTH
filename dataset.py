import os
import torch
import numpy as np
import pickle
from collections import Counter
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from conf import conf

# 获取数据集，如果是服务器直接返回loader
def get_dataset(ifserver):
    dir=conf['dataset_dir']
    name=conf['dataset_name']
        # download=true表示从下载数据集并把数据集放在root路径中
    if name=='mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=trans_mnist)
        eval_dataset = datasets.MNIST(dir, train=False, download=True, transform=trans_mnist)

    elif name=='cifar10':
            # transform：图像类型的转换
        # 用Compose串联多个transform操作
        transform_train = transforms.Compose([
                # 四周填充0，图像随机裁剪成32*32
            transforms.RandomCrop(32, padding=4),
                # 图像一半概率翻转，一半概率不翻转
            transforms.RandomHorizontalFlip(),
                # 将图片(Image)转成Tensor，归一化至[0, 1]
            transforms.ToTensor(),
                # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
                # 将图片(Image)转成Tensor，归一化至[0, 1]
            transforms.ToTensor(),
                # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

            #得到训练集
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                        transform=transform_train)
            #得到测试集
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    else:
        print('Wrong dataset name')

    #如果是服务器 直接返回测试集用来测试全局模型的精度
    if ifserver:
        train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"])
        eval_loader=torch.utils.data.DataLoader(eval_dataset, batch_size=conf["batch_size"])
        return train_loader,eval_loader
    else:
        return train_dataset,eval_dataset

#返回所有客户端的训练/测试样本列表和数据分布    
def get_data_indices(train_dataset,eval_dataset):
    train_indices_list=[]
    eval_indices_list=[]
    train_data_dis=[]
    if conf['iid']==True:
        # 按客户端id划分数据集 随机抽取样本（iid）
        train_range = list(range(len(train_dataset)))
        eval_range = list(range(len(eval_dataset)))
        # train_data_len是每个客户端的数据量
        train_data_len = int(len(train_dataset) / conf['num_client'])
        eval_data_len = int(len(eval_dataset) / conf['num_client'])
        # 根据客户端的id来平均划分训练集和测试集，indices为该id下的子训练集
        for id in range(conf['num_client']):
            train_indices_list.append(train_range[id * train_data_len: (id + 1) * train_data_len])
            eval_indices_list.append(eval_range[id * eval_data_len: (id + 1) * eval_data_len]) 

        
    else:
        if conf['dataset_name']=='cifar10': 
            #noniid用main中生成好的数据idx生成数据集
            with open('noniid/cifar10_train.pkl','rb') as f:
                train_indices_list=pickle.load(f)
                train_indices_list=list(train_indices_list.values())
            with open('noniid/cifar10_test.pkl','rb') as f:
                eval_indices_list=pickle.load(f)
                eval_indices_list=list(eval_indices_list.values())
        elif conf['dataset_name']=='mnist':
            with open('noniid/mnist_train.pkl','rb') as f:
                train_indices_list=pickle.load(f)
                train_indices_list=list(train_indices_list.values())
            with open('noniid/mnist_test.pkl','rb') as f:
                eval_indices_list=pickle.load(f)
                eval_indices_list=list(eval_indices_list.values())
            
    #统计各类数据数量分布的函数，得到一个list
    for indices in train_indices_list:
        subdata = Subset(train_dataset,indices)
        train_data_dis.append(dis_total(subdata))
    del subdata
    return train_indices_list, eval_indices_list,train_data_dis


#从dataloader统计数据分布,num_class是数据集总类别数量
def dis_total(dataset,num_class=conf['num_class']):
    datasize=len(dataset)
    loader=DataLoader(dataset,batch_size=128,shuffle=False)
    label_counter=Counter()
    for _, labels in loader:
    # 更新计数器
        label_counter.update(labels.tolist())  # 假设 labels 是一个 torch 张量  
    label_list=[0]*num_class
    for label, count in label_counter.items():
        label_list[label]=count/datasize
    return label_list
   

# 以下用于划分noniid数据集
def cifar_extr_noniid(train_dataset, test_dataset, num_users = conf['num_client'], n_class=conf['n_class'], num_samples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    num_shards_train, num_imgs_train = int(50000/num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)} 
    idxs = np.arange(num_shards_train*num_imgs_train)
    # labels = dataset.train_labels.numpy()
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)
    #labels_test_raw = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]
    #print(idxs_labels_test[1, :])


    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand*num_imgs_train:int((rand+rate_unbalance)*num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        #print(user_labels_set)
        #print(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label)*num_imgs_perc_test:int(label+1)*num_imgs_perc_test]), axis=0)   
        #print(set(labels_test_raw[dict_users_test[i].astype(int)]))
        dict_users_train[i]=dict_users_train[i].astype(int)
        
        dict_users_test[i]=dict_users_test[i].astype(int)
        

    return dict_users_train, dict_users_test

def get_dataset_cifar10_extr_noniid(num_users = conf['num_client'], n_class=conf['n_class'], nsamples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    data_dir = conf['dataset_dir']
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test


# MNIST 10 classes 70000 pics, 60000 for training and 10000 for test
def mnist_extr_noniid(train_dataset, test_dataset, num_users=conf['num_client'], n_class=conf['n_class'],
                      num_samples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    num_shards_train, num_imgs_train = int(60000 / num_samples), num_samples
    num_classes = 10
    num_imgs_perc_test, num_imgs_test_total = 1000, 10000
    assert(n_class * num_users <= num_shards_train)
    assert(n_class <= num_classes)
    idx_class = [i for i in range(num_classes)]
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    dict_users_test = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)
    idxs_test = np.arange(num_imgs_test_total)
    labels_test = np.array(test_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:(rand + 1) * num_imgs_train]), axis=0)
            else:
                dict_users_train[i] = np.concatenate(
                    (dict_users_train[i], idxs[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
                user_labels = np.concatenate((user_labels, labels[rand * num_imgs_train:int((rand + rate_unbalance) * num_imgs_train)]), axis=0)
            unbalance_flag = 1
        user_labels_set = set(user_labels)
        for label in user_labels_set:
            dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test[int(label) * num_imgs_perc_test:int(label + 1) * num_imgs_perc_test]), axis=0)
        dict_users_train[i] = dict_users_train[i].astype(int)
        dict_users_test[i] = dict_users_test[i].astype(int)

    return dict_users_train, dict_users_test

def get_dataset_mnist_extr_noniid(num_users=conf['num_client'], n_class=conf['n_class'], nsamples=conf['nsamples'], rate_unbalance=conf['rate_unbalance']):
    data_dir = conf['dataset_dir']
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,transform=apply_transform)
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def mnist_noniid(dataset, num_users):
    num_imgs = len(dataset) // num_users // 2
    num_shards = num_users * 2
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels[:num_shards * num_imgs]))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__=='__main__':
#    生成客户端noniid数据分布，共100客户端，每个客户端2个类别共400个样本。
    # _,_,user_groups_train, user_groups_test=get_dataset_cifar10_extr_noniid(num_users =100, n_class=2, nsamples=200, rate_unbalance=1)
    # with open('noniid\cifar10_train.pkl','wb') as f:
    #   pickle.dump(user_groups_train,f)
    # with open('noniid\cifar10_test.pkl','wb') as f:
    #   pickle.dump(user_groups_test,f)


    # train_data, test_data = get_dataset(False)
    # train_indices, eval_indices,train_dis=get_data_indices(train_data,test_data)
    
    # download and load the dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    test_dataset = datasets.MNIST('../data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    # create the dictionaries
    # mnist_100clients_train = mnist_noniid(train_dataset, 100)
    # with open('noniid/mnist_100clients_train.pkl','wb') as f:
    #   pickle.dump(mnist_100clients_train,f)
    # mnist_100clients_test = mnist_noniid(test_dataset, 100)
    # with open('noniid/mnist_100clients_test.pkl','wb') as f:
    #   pickle.dump(mnist_100clients_test,f)
    
     
    from fedlab.utils.dataset import MNISTPartitioner
    user_dict = MNISTPartitioner(train_dataset.targets, 
                                        num_clients=conf['num_client'],
                                        partition="iid" if conf['iid'] else "noniid-labeldir", 
                                        dir_alpha=conf['dir_alpha'],
                                        seed=1234,verbose=False).client_dict
    with open('noniid/mnist_train.pkl','wb') as f:
        pickle.dump(user_dict,f)
    user_dict = MNISTPartitioner(test_dataset.targets, 
                                        num_clients=conf['num_client'],
                                        partition="iid" if conf['iid'] else "noniid-labeldir", 
                                        dir_alpha=conf['dir_alpha'],
                                        seed=1234,verbose=False).client_dict
    with open('noniid/mnist_test.pkl','wb') as f:
        pickle.dump(user_dict,f)
