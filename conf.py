# @zhy
# fedlth的设置模块
conf={
    #服务器IP地址
#   222.20.126.150
    "ip":"127.0.0.1",
    #服务监听端口
    "port":5003,

    #总客户端数量
    "num_client" :100,
    # 一个client脚本中模拟出的客户端数量
    "v_client":100,

    #全局迭代次数：即服务端和客户端的通信次数
    #通常会设置一个最大的全局迭代次数
	"global_epoch" : 100,

    #本地模型的迭代次数：即每一个客户端在进行本地模型训练时的迭代次数
	"local_epoch" : 2,
    #本地模型进行训练时的参数-每个batch的大小
	"batch_size" : 32,

    #数据集名称
    # "dataset_name":'cifar10',/
     "dataset_name" :'mnist',
    # 数据集类别数量
    'num_class':10,
    #数据集位置
    "dataset_dir":'../data',
    "use_sparse_conv" : False,

    #初始模型的文件路径：
    # "pretrained" : "pretrained/1checkpoint.pth.tar",
    # "init_model":'models/init_cnn_cifar.pt',
    # "init_model": 'models/init_cnn_mnist.pth',
    # "init_model": 'models/init_resnet18.pt',
    # 临时文件路径 存放训练过程发送/接收的数据
    "temp_path":"temp",
    # 开始剪枝的精度
    'start_prune':0.7,


    # 是否iid
    'iid':True,
    #狄利克雷分布设置： 在分割数据集的dataset.py的main函数中使用
    'dir_alpha':0.3,
    #noniid设置： 在分割数据集的dataset.py的main函数中使用
    # 每个客户端类别数量
    'n_class':3,
    # 每类别数量
    'nsamples':200,
    # 分割后每个客户端的数据集数量是 n_class* nsamples
    # 不平衡度
    'rate_unbalance':1.0,



    #每次选取k比例客户端参与迭代
	"k" : 0.2,



    #本地模型进行训练时的参数-学习率
	"lr" : 0.01,
    "min_lr": 0.005,
    "decrease_rate": 0.1,
    "decrease_frequency": 10,    # epoch
    "decreasing_lr" : "80,120",
    #本地模型进行训练时的参数-momentum
	"momentum" : 0.0001,
	#weight decay
    "weight_decay" : "0.0001",

    #本地模型进行训练时的参数-正则化参数
	"lambda" : 1,

    "prune_type" : "lt",
    "start_ratio" : 0.1,

    #Global_model设备
    "global_dev" : 'cuda',
    'local_dev':'cuda',


    #近似差分隐私delta：容错率
    "delta": 1e-5,
    #裁剪l2_norm_clip =0.1 ; None(中位数裁剪)
    "clip":0.13,
    #noise_multiplier=2 ; 0(不使用差分隐私)
    "noise_multiplier":0.5,
    #差分隐私预算
    "epsilon":10
}
