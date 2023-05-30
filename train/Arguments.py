import os


class Arguments:
    def __init__(self, *args):
        # env related
        self.gamma = 0.9
        self.reward_type = "bce"
        self.ips = True

        # network related
        self.embed_dim = 128
        self.drop_out = 0.2
        self.ou_noise_theta = 0
        self.ou_noise_gamma = 0.4
        self.actor_lr = 1e-3  # 学习速度偏慢
        self.critic_lr = 1e-3
        self.actor_reg = 0  # 每个模型都有个无底洞的reg rate要调整，从1到0的顺序吧
        self.actor_update_freq = 1  # critic学完后完全可以直接每步都更新
        self.soft_update_freq = 2
        self.tau = 0.2

        # training related
        self.epoch = 1000
        self.batch_size = 2048
        self.memory_size = 500000
        self.init_episode = 10000   # 初始采样轮数
        self.init_training_step = 10000  # 初始Critic训练步数

        self.train_rows = 100000
        self.test_rows = 50000

        # path related
        self.test_path = "./dataset/rt/test.csv"
        self.val_path = "./dataset/rt/val.csv"
        self.train_path = "./dataset/rt/train.csv"
        self.features_path = "./dataset/rt/item_feadf.csv"
        self.map_path = "./pretrain"
        self.pretrain_path = "./pretrain/"
        self.memory_path = "./pretrain/memory.pkl"

    def set_curpath(self, model_name):
        self.cur_path = f"./chkpt/RL/res_{model_name}"
        if not os.path.isdir(self.cur_path):
            os.makedirs(self.cur_path)
        self.save_dir = self.cur_path
