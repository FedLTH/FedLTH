import socket
from conf import conf
from global_model import Global_model
from models.cnn_cifar import CNNCifar
from models.cnn_mnist import CNNMNIST
from pruning_utils import *
import os,struct,pickle
from chooseclient import client_group
import torch
import random
from gzip import compress,decompress
import copy
import rdp

#发送文件
def send_file(sock,file):
	with open(file,'rb') as f:
		data=f.read()
	send_data(sock,data)

# 发送数据函数     # sock:接收方socket
def send_data(sock, data):
	# 序列化数据
	data_bytes = compress(pickle.dumps(data))
	# 发送数据大小
	data_size = len(data_bytes)
	sock.sendall(struct.pack('>I', data_size))
	# 发送数据内容
	sock.sendall(data_bytes)
	# # 接收状态码，确定是否成功
	code = struct.unpack(">I", sock.recv(4))[0]
	if code!=200:
		raise Exception(f"send data error,socket: {sock}")

# 接收数据函数     # sock:发送方socket
def recv_data(sock):
	msg_len = struct.unpack(">I", sock.recv(4))[0]
	msg = sock.recv(msg_len, socket.MSG_WAITALL)
	msg = pickle.loads(decompress(msg))
	sock.sendall(struct.pack('>I', 200))
	# 表示成功接收
	return msg



#建立连接
def conn():
	listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	listening_sock.bind((conf['ip'], conf['port']))
	client_sock_all = []

	# Establish connections to each client, up to n_nodes clients
	while len(client_sock_all) < conf['num_client']:
		listening_sock.listen(5)
		print("Waiting for incoming connections...")
		(client_sock, (ip, port)) = listening_sock.accept()
		print('Got connection from ', (ip, port))
		client_sock_all.append(client_sock)
		# 下发id， 从0开始
		id= len(client_sock_all)-1
		send_data(client_sock,id)
	return listening_sock,client_sock_all

# 联邦训练函数
def fed_train(part_id,global_model,lr=conf['lr'],noise_multiplier=0):
	if global_model.global_epoch>=conf['global_epoch']:
		# 如果global_epoch达到直接返回
		return
	#获取客户端id对应的socket对象，需要训练的发送train命令，不参与的发送wait
	client_update=[]
	client_acc=[]
	client_loss=[]
	max_client_traintime=0
	upload_size=[]
	down_size=len(compress(pickle.dumps(global_model.model.state_dict())))
	for id in range(len(clients)):
		sock=clients[id]
		# 下发全局轮数
		if id in part_id:
			# 该客户端参与训练
			# 下发训练命令
			send_data(sock,'train')
			send_data(sock,global_model.global_epoch)
			# 下发全局模型、学习率和隐私参数
			send_data(sock,[global_model.model.state_dict(),lr,noise_multiplier])
			# 接收更新信息
			data=recv_data(sock)
			client_update.append(copy.deepcopy(data[0]))
			client_loss.append(data[1])
			client_acc.append(data[2])
			# client_traintime+=data[3]
			if max_client_traintime< data[3]:
				max_client_traintime=data[3]
			upload_size.append(len(compress(pickle.dumps(data))))
		else:
			# 该客户端等待
			send_data(sock,'wait')
			send_data(sock,global_model.global_epoch)
	#接收到全部更新,开始聚合
	global_model.aggregate(client_update,client_loss,client_acc)
	acc_global.append(global_model.eval())
	round_comm_size=sum(upload_size)+len(part_id)*down_size
	# client_traintime=client_traintime/len(part_id)
	comm_datasize_list.append(round_comm_size)
	client_traintime_list.append(max_client_traintime)
	print(f'Round:{global_model.global_epoch}; Acc:{acc_global[-1]*100:.3f},Total_comm_size:{round_comm_size}')
	print(f'Server->Client:{down_size} ; Client->Server:{upload_size} ' )
	# torch.save(global_model,os.path.join(conf['temp_path'],f'global{global_model.global_epoch}_model'))

class LR_UPDATE():
	def __init__(self,init_lr=conf['lr'],
			  decrease_rate=conf['decrease_rate'],
			  min_lr=conf['min_lr'],
			  decrease_frequency:int =conf['decrease_frequency']):
		self.counter = 0
		self.init_lr = init_lr
		self.lr = init_lr
		self.decrease_rate = decrease_rate
		self.min_lr = min_lr
		self.decrease_frequency = decrease_frequency

	def __call__(self, reset=False):
		self.counter += 1
		if reset:
			self.lr = self.init_lr
			self.counter = 0
			return self.lr
		if self.counter % self.decrease_frequency == 0:
			self.lr = max(self.min_lr, self.lr * (1.-self.decrease_rate))
		return self.lr

if __name__=='__main__':
  # 初始化
	global_model=Global_model()
	#记录当前连接的客户端
	clients=[]

	# 保存客户端信息
	client_info=dict()
	#初始客户端组的id
	group_id=0

  # 保存客户端更新
	client_update=[]
  # 保存客户端分组
	groups=[]
	#记录发生剪枝的轮数
	prune_round=[]
	# 记录通信流量
	comm_datasize_list=[]
	# 记录平均训练时间
	client_traintime_list=[]
	#记录全局模型精度
	acc_global=[]




	# 建立连接、下发id和初始化模型
	server,clients=conn()
	print('all client connected')
	# print(f'model:{conf["init_model"]}')
	print(f'dataset:{conf["dataset_name"]}')
	print(f'num_client:{conf["num_client"]}')
	print(f'dp-epsilon:{conf["epsilon"]}')

	# 分组过程
	for client in clients:
		# 发送分组命令和初始模型
		send_data(client,'group')
		send_data(client,global_model.model)
		# 接收客户端信息
		data=recv_data(client)
		client_info[data['id']]=data
		# print(f'recv info client{data["id"]}')
#单个客户端信息结构
# info= {'id':客户端id
# 'data_dis':list,数据分布
# 'train_data_len':number, 训练集大小
# 'train_time':number, 训练时间
# 'prune_ratio':number 剪枝率，默认是0}
  # 将平均值作为训练时间阈值
	time_list=[client_info[id]['train_time']for id in client_info]
	avgtime=sum(time_list)/len(time_list)
	print(f'Client train Avgtime:{avgtime}')
	#广播时间阈值
  #等待客户端返回剪枝率
	for sock in clients:
		send_data(sock,avgtime)
		data=recv_data(sock)
		#data[0]客户端id; data[1] 参数稀疏率  data[2]通道稀疏率，tp剪枝用
		client_info[data[0]]['prune_ratio']=data[1]
		client_info[data[0]]['channel_sparsity']=data[2]
		# print(f'recv client{data[0]},prune_ratio:{data[1]}')

	# # 保存客户端信息
	# with open('result/client_info100.pkl','wb') as f:
	# 	pickle.dump(client_info,f)
	# 测试代码
	# with open('result/client_info100.pkl','rb') as f:
	# 	client_info=pickle.load(f)
	# # 开始模拟退火分组
	# 	# 接收完客户端信息 开始客户端分组
		# 20个客户端信息
	# with open('result/client_info20.pkl','rb') as f:
	# 	client_info=pickle.load(f)

	# client_info[0]=  {'id': 1, 'data_dis': [0.1024, 0.106, 0.0956, 0.1004, 0.0908, 0.0984, 0.1024, 0.1044, 0.0996, 0.1], 'train_data_len': 2500, 'train_time': 12.279283255338669, 'prune_ratio': 0.47, 'channel_sparsity': 0.1}
	# client_info[1]=  {'id': 2, 'data_dis': [0.1024, 0.106, 0.0956, 0.1004, 0.0908, 0.0984, 0.1024, 0.1044, 0.0996, 0.1], 'train_data_len': 2500, 'train_time': 12.279283255338669, 'prune_ratio': 0.6179542324821345, 'channel_sparsity': 0.15}
	groups=client_group(client_info=client_info)

	# 前2个组剪枝率设为0，后3个组设为0.1
	for group_idx in range(len(groups)):
		if group_idx < 2:
			prune_ratio = 0.0
			channel_sparsity = 0.0
		else:
			prune_ratio = 0.1
			channel_sparsity = 0.1

		for client_id in groups[group_idx]:
			client_info[client_id]['prune_ratio'] = prune_ratio
			client_info[client_id]['channel_sparsity'] = channel_sparsity

	# groups=[[0],[1]]#测试用
	group_id=0
	print(f'group finish,groups:{groups}')
	# 初始化差分隐私参数
	noise_multiplier=conf['noise_multiplier']
	if conf['noise_multiplier']!=0:
		# noise_multiplier=rdp.get_noise_multiplier([conf["noise_multiplier"]],[0],conf,600,conf['global_epoch']/conf['k'],target_epsilon=conf["epsilon"])
		noise_multiplier=conf['noise_multiplier']
		print(f'Init noise:{noise_multiplier}')
	lr_update = LR_UPDATE()

	#创建早期重置点rewind_weight 应当在训练前被定义
	while global_model.global_epoch<conf['global_epoch']:
		fed_train(range(len(clients)), global_model)
		if global_model.acc[-1]>conf['start_prune']:
			rewind_weight = copy.deepcopy(global_model.model.state_dict())
			break

	# 开始联邦剪枝过程
	print('start fed prune')
	while group_id<len(groups) and global_model.global_epoch<conf['global_epoch']:
		# 每次结构剪枝并切换客户端组后下发模型结构
		for client in clients:
			send_data(client,'model')
			send_data(client,global_model.model)
		# 精度50后开始剪枝
		while global_model.acc[-1]<conf['start_prune'] and global_model.global_epoch<conf['global_epoch']:
			fed_train(groups[group_id], global_model,lr=lr_update())
		#剪枝间隔轮数
		prune_step=3
		u_pruned_flag=False
		weight_with_mask = global_model.model.state_dict()
		global_model.init_ratio()
		target_ratio = max([client_info[id]['prune_ratio'] for id in groups[group_id]])
		channel_sparsity = max([client_info[id]['channel_sparsity'] for id in groups[group_id]])
		#测试用例Target_ratio
		while global_model.global_epoch< conf['global_epoch']:
			#训练
			fed_train(groups[group_id], global_model,lr=lr_update())
			#每过prune_step轮触发一次剪枝
			if global_model.global_epoch % prune_step == 0:
				if global_model.acc[-1]<conf['start_prune']:
					print('skip un_prune')
					continue

				print(f'Unstruct Prune, Prune Ratio: {global_model.ratio},Target Ratio:{target_ratio}')
				u_pruned_flag=True
				prune_round.append(global_model.global_epoch)
				# 非结构化剪枝（可迭代）
				global_model.u_prune(global_model.ratio)
				weight_with_mask = global_model.model.state_dict()
				remove_prune(global_model.model, conv1=False)
				# 如果达到目标剪枝率，跳出循环
				if global_model.ratio >= target_ratio:
					print('Reach Target Ratio')
					break
				global_model.increase_ratio(target_ratio)
			#剪枝后的一轮，如果精度下降超过5%则增大剪枝间隔
			if global_model.global_epoch % prune_step == 1:
				if global_model.acc[-2]-global_model.acc[-1]>0.05:
					prune_step+=1
		# 结构化剪枝重组
		# 如果经过非结构化剪枝才可以refill
		if u_pruned_flag==False:
			continue
		prune_mask = global_model.refill(weight_with_mask, mask_only=True)
		# remove_prune(global_model.model, conv1=False)
		# Recover weight
		global_model.model.load_state_dict(rewind_weight)
		# Apply Sparity to model
		prune_model_custom(global_model.model, prune_mask, conv1=False)
		remove_prune(global_model.model, conv1=False)
		u_pruned_flag=False
		# TP Permenant Prune
		global_model.model.zero_grad()
		trace_input, _ = next(iter(global_model.train_loader))
		# Reset Learning rate
		lr_update(reset=True)
		######################把这里的0.3替换为通道剪枝率#######################
		print('Refill Struct Prune.Final sparsity:' + str(100 *
									  global_model.tp_prune(trace_input.to(global_model.device),
															channel_sparsity, imp_strategy='Magnitude',
															degree=1) # type: ignore
									  ) + '%')
		rewind_weight=copy.deepcopy(global_model.model.state_dict())
		#切换客户端组
		group_id+=1
		# torch.save(global_model.model,f'temp/resnet{global_model.global_epoch}.pth')
	print('fed prune finish')

	  # 剪枝结束，微调
		# 如果还有剩余轮次可以微调，先下发最终的模型结构
	if	global_model.global_epoch<conf['global_epoch']:
		for client in clients:
			send_data(client,'model')
			send_data(client,global_model.model)
	# 重置差分隐私预算
	# 计算剩余轮数和差分隐私噪声强度
	if conf['noise_multiplier']!=0:
		remaining_epochs = conf['global_epoch'] - global_model.global_epoch
		noise_multiplier=rdp.get_noise_multiplier([conf["noise_multiplier"]],[0],conf,60000,remaining_epochs/conf["k"],target_epsilon=conf["epsilon"])
	print(f'fine tune noise:{noise_multiplier}')
	# 按k比例抽取客户端微调训练
	global_model.model.load_state_dict(rewind_weight)
	lr_update(reset=True)
	while global_model.global_epoch<conf['global_epoch']:
		ids=random.sample(range(len(clients)), int(conf['k']*len(clients)))
		fed_train(ids, global_model,lr=lr_update(),noise_multiplier=noise_multiplier)
		# group=random.sample(groups , 1)[0]
		# fed_train(group, global_model)

	# 微调结束，全局模型评估
	test_acc=global_model.eval()

	eval_info=[]
	for sock in clients:
		# 下发指令
		send_data(sock,'eval')
		send_data(sock,global_model.model)
  #等待客户端返回统计数据
		data=recv_data(sock)
		eval_info.append(data)
		# print(f'recv info from client{data["id"]}')

	hpara='fedlth_{}_g{}_l_{}_k{}_n{}_iid{}.pkl'.format(conf['dataset_name'], conf['global_epoch'], conf['local_epoch'], conf['k'], conf['num_client'],conf['iid'])
	server_eval={
		'global_acc':acc_global,
		'train_acc':global_model.acc,
		'prune_round':prune_round,
		'comm_size':comm_datasize_list,
		'train_time':client_traintime_list
	}

	with open(f'result/client_info_{hpara}.pkl','wb') as f:
		pickle.dump(eval_info,f)
	with open(f'result/global_eval_{hpara}.pkl','wb') as f:
		pickle.dump(server_eval,f)

	print(f'FL done. Test_Acc:{test_acc*100:.2f}; Comm_Size:{sum(comm_datasize_list)}')
