# rlmtlrec code structure
## 模型代码
+ layers: 储存常用网络结构
  + critic: critic网络
  + esmm: esmm(actor)网络，可以在slmodels里面引入其他MTL模型作为actor
  + layers: 经典Embedding层和MLP层
+ slmodels: SL baseline模型
+ agents: RL模型
+ train: 训练相关配置
+ env.py: offline采样模拟环境
+ RLmain.py: RL训练主程序
+ SLmain.py: SL训练主程序

## 读写代码
+ dataset
  + rtrl：retrailrocket数据集（组织成MDP格式：）[timestamp,sessionid,itemid,pay,click], [itemid,feature1,feature2,..],6:2:2
    
+ chkpt
+ pretrain


## layers
+ layers:
  + EmbeddingLayer(field_dims, embed_dim) 
  + MultiLayerPerceptron(input_dim, embed_dims, dropout, output_layer=True)
    
+ critic
    + Critic(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims,
                 dropout): 
        + 参数说明：
          + categorical_field_dims, numerical_num 分别表示类别特征维度（对应embedding）和数值特征维度
          + embed_dim, bottom_mlp_dims, tower_mlp_dims 为网络结构参数，一般和baseline选择同一套参数（后续可调）
          + dropout 默认值为0.2，可调
        + 结构：类别特征、数值特征以及action（1维）的embedding拼接，输入bottom_mlp层再经过tower_mlp层输出1维的Critic打分q值
        + forward(self, categorical_x, numerical_x, action) -> 1维长度为1的tensor
    + CriticNeg(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims,
                 dropout): 
        + 结构同上，改激活函数为ReLU，使得forward输出为负值

+ esmm
    + ESMM(categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num,
                 dropout)
       + 参数说明：
          + task_num: 任务数量（本paper设定为2，只讨论CTR和CTCVR任务）
       + 结构：类别特征、数值特征的embedding拼接输入到相同的bottom_mlp层，然后从task_num个tower_mlp输出1维预测值
       + forward(self, categorical_x, numerical_x) -> 1维长度为task_num的tensor

## slmodels
直接搬用开源的MTL模型整合包，[链接](https://github.com/easezyc/Multitask-Recommendation-Library.git)
，均为上述ESMM结构：输入输出一致，网络结构参数有部分不同，不过都预设好了


## SLmain.py:
### 导入模块：
+ DL基础包
+ 数据(例): from dataset.rtrl import RetailRocketRLDataset
+ MTL模型: slmodels目录
+ RL相关: 用于二阶段训练

### 函数：
+ 需要从train.run中导入的训练环境函数：
    + get_dataset(name, path)
    + get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim)
    + sltrain(model, optimizer, data_loader, criterion, device, polisher=None, log_interval=100)
      + 返回epoch_loss
    + sltest(model, data_loader, task_num, device)
      + 返回测试auc和logloss
    + slpred(model, data_loader, task_num, device)
      + 返回多目标预测值
+ main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         feature_map_rate,
         batch_size,
         embed_dim,
         weight_decay,
         polish_lambda,
         device,
         save_dir)
  + 参数说明：
    + 数据加载参数：dataset_name, dataset_path, feature_map_rate （防止类别标签爆炸）
    + 模型参数: task_num, expert_num, model_name
    + 常见训练可调节参数：epoch,learning_rate,batch_size,embed_dim,weight_decay 
    + **本模型特色参数**：polish_lambda （将新权重以一定比率添加到原来Loss）
    + 外部参数：save_dir, device
### 运行实例
python3 SLmain.py --model_name=esmm
python3 RLmain.py
python3 SLmain.py --model_name=ple --polish=1

返回情况：

test: best auc: 0.732444172986328
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 134/134 [00:07<00:00, 19.14it/s]
task 0, AUC 0.7273702846096346, Log-loss 0.20675417715656488
task 1, AUC 0.7247954179346048, Log-loss 0.048957254763240504

同时chkpt文件新增 .pt的最优模型储存文件

## agents
### ReplayBuffer.py
+ ReplayBuffer
    + init(state_dim, action_dim, size, batch_size): 生成[state, action, next_state, reward, done, label]的buffer
    + store: 将探索产生的transition存到buffer
    + sample_batch: 按照batch_size不重复地从buffer采样
    
### DDPG_ESMM类继承体系
+ DDPG_ESMM
    + init:
      参数表格:
      
      |参数名|用途|默认值|
      |---|---|---|
      |env| 环境类 | 无 |
      |actor_name|MTL模型名|"esmm"|
      |gamma|discount|0.9|
      |pretrain_path|预训练地址|"./pretrain"|
    
      |参数名|用途|默认值|
      |---|---|---|
      |**embed_dim**|同SL|128|
      |bottom_mlp_dims|共享层维度|(512,256)|
      |tower_mlp_dims|任务层维度|(128,64)|
      |ou_noise_theta|噪声均值|0|
      |ou_noise_gamma|噪声方差|0.4|
      |memory_size|replay buffer大小|500000|
      |actor_reg|BC权重|0|
      |tau|软更新比例|0.2|
      |soft_update_freq|软更新频率|2|
      |actor_update_freq|actor更新频率|2|
      |init_training_step|初始训练步数|10000|
      |*ips*|ips样本权重|False|
      |batch_size| |512|
      |drop_out| |0.2|
      |actor_lr| |1e-3|
      |critic_lr| |1e-3|
      


+ DDPG_ESMM
    + init，初始化7个网络，具体参数参考上表:
        + pretrain_actor: 预训练的MTL模型
        + actor, actor_target 
        + critic1/2, critic1/2_target
    + select_action(self, state: np.ndarray) -> np.ndarray
    + process_batch(self, transition: Dict[np.ndarry,...]) -> Dict[torch.Tensor,...]
    + get_closs(self, critic_id: int, critic: nn.Module, critic_target: nn.Module, transition: Dict) -> torch.Tensor
    + get_aloss(self, transition: Dict) -> Tuple[torch.Tensor,torch.Tensor]
    + update(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
      + 正常DDPG更新逻辑，注意init_training_step后再更新actor网络
    + _target_soft_update(self, tau:float) 
    + save_or_load_agent(self, cwd: str, if_save: bool)
    
+ DDPG_ESMM_BC(DDPG_ESMM_multiweight)
    + 针对offline数据进行训练，在process_batch,get_closs,get_aloss按照TD3BC的修改方式进行了修改，同时update只更新Critic网络至收敛

## train
+ Arguments
    + 参数类：模型参数，训练参数，文件参数
    
+ utils 工具函数
    + get_optim_param: 固定用于存储模型
    + Catemapper: 大量类别特征会导致embedding爆炸，此处会设定一个阈值将长尾的类别都判定为1个类（需要对不同数据集调整）
    + EarlyStopper: SL模型停止条件，验证集指标连续数次不提升就停止
    + ConvergeStopper: （原创）RL的loss连续多次变化小于一定范围就判定为收敛，随后终止训练保存模型
    + ActionNormalizer: 环境状态标准化程序，gym常见，可作为component调节项
    + RlLossPolisher: 我们的模型，根据Critic的分值给出各个item在不同目标下的权重，权重组合方式可调
    
+ run 训练整合（此处仅描述RL相关）
    + one_iter_offline(agent): 用agent采样原始数据集一次，产生transition存入replaybuffer
    + train_and_test_offline(environment: gym.Env, agent, epoch, init_episode, save_dir, memory_path)
    + test(environment, agent, save_dir): 可以使用测试环境查看动态RL训练效果，本模型用不上
    + plot_loss: 绘制RL训练过程中score，critic_loss，actor_loss以及actor_loss_reg的变化曲线（可以进一步修改）

## RLmain.py
+ create_sub_agent(env, actor_name, agentcls, hyparams)：定义agent

## 全部实验训练流程