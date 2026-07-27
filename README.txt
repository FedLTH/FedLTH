FedLTH: A Privacy-preserving Federated Learning Framework with Model Pruning on Edge Clients
Abstract: Although Federated Learning (FL) enables distributed clients to cooperatively train deep learning models without sharing local data, the iterative FL training process imposes considerable computation and communication overheads on clients. Especially in cloud-edge collaboration situations, heterogeneous and resource-limited edge clients can become a bottleneck for FL. In this paper, we propose FedLTH (Federated Learning with the Lottery Ticket Hypothesis), an FL framework based on the Lottery Ticket Hypothesis and adaptive differential privacy, which aims to improve communication and computing efficiency and privacy security for resource-limited edge clients. First, the pruning rate of each client is set according to their respective resource constraints. The server divides the clients into groups with balanced data distribution and similar pruning rates to ensure the convergence of the global model. Then, a structured model pruning method based on the Lottery Ticket Hypothesis is introduced. Each client group participates in a pruning phase to reduce the computing overhead of clients. Last, an adaptive differential privacy algorithm is designed to preserve client data privacy and improve model accuracy. Through experiments on multiple datasets and non-IID scenarios, we show the effectiveness of FedLTH in privacy preservation and reducing computation and communication overheads

设置： conf.py中 客户端数量、差分隐私、iid选项、vclient 一个client脚本模拟的客户端数量
启动：
server.py
client.py

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install torch-pruning==1.3.6  matplotlib numpy prv-accountant fedlab
