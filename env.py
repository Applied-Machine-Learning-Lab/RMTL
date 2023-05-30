# mtrec env
import gym
import pandas as pd
import numpy as np
from collections import defaultdict
from train.utils import Catemapper

class Visitor:
    def __init__(self, session, reward_type="norm"):
        self.session = session
        self.reward_type = reward_type
        self.states = session['states']
        self.state = self.states[0]  # initial state

        self.labels = session['labels']

        self.session_len = min(self.states.shape[0],self.labels.shape[0])
        self.timestep = 0

    def step(self, action):
        t = self.timestep
        label = self.labels[t]
        cvaction = np.array([action[0], action[1]])          # CTR/CTCVR accu
        # cvaction = np.array([action[0], action[0]*action[1]])  # CTR/CVR accu

        # TODO: weighted reward
        # reward = np.clip(1 - abs(cvaction - label), 0, 1)
        reward = -np.abs(cvaction - label)

        # BCE reward
        if self.reward_type == "bce":
            reward = label*np.log(np.clip(cvaction,1e-4,1))+(1-label)*np.log(np.clip(1-cvaction,1e-4,1))

        # print(action,cvaction, reward)
        if t + 1 < self.session_len:
            nstate = self.states[t + 1]
            done = False
        else:
            nstate = self.states[t]
            done = True
        self.timestep += 1
        return nstate, reward, done, label

    def __len__(self):
        return self.labels.shape[0]

    def __str__(self):
        print(self.session)

class seqVisitor(Visitor):
    def __init__(self, session):
        super(seqVisitor, self).__init__(session)
        self.session = session
        self.states = session['states']
        self.state = self.states[[0]]  # initial state

        self.labels = session['labels']

        self.session_len = self.labels.shape[0]
        self.timestep = 0

    def step(self, action):
        t = self.timestep
        label = self.labels[t]
        # 注意这里要改，或者额外弄个环境
        cvaction = np.array([action[0], action[0]*action[1]])  # CTR/CTCVR accu
        # TODO: sigle add 模型不需要限制范围
        reward = np.clip(1 - abs(cvaction - label), 0, 1)
        # print(action,cvaction, reward)
        if t + 1 < self.session_len:
            nstate = self.states[:(t + 2),:]
            done = False
        else:
            nstate = self.states[:(t + 1),:]
            done = True
        self.timestep += 1
        return nstate, reward, done, label



# class MTEnv(gym.Env):
class MTEnv(gym.Env):
    def __init__(self, mdp_path, features_path, map_path, nrows=10000,reward_type="norm",is_test=False,is_seq=False):
        super(MTEnv, self).__init__()
        self.mdp_path = mdp_path
        self.nrows = nrows
        self.features_dict, self.idmap = self.get_features(features_path, map_path)
        self.field_dims = self.idmap.field_dims

        self.is_test = is_test
        self.reward_type = reward_type
        self.test_step = 0
        self.is_seq = is_seq

        self.action_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)  # 2维0-1, sample...
        # self.observation_space = gym.spaces.Discrete() # t*feature_len, dynamic; hard to represent here

    def get_features(self, features_path, map_path):
        feature_cols = ['785', '591', '814', 'available', 'categoryid', '364', '776']
        features = pd.read_csv(features_path, usecols=feature_cols + ['itemid'])
        features.drop_duplicates('itemid', inplace=True)
        features.fillna(0, inplace=True)
        idmap = Catemapper(threshold=0.2)
        idmap.load_mapper(map_path)
        idmap.map(features)
        features_dict = dict(zip(features['itemid'].tolist(), features[feature_cols].values))
        return features_dict, idmap


    def getMDP(self):
        mdp_data = pd.read_csv(self.mdp_path, usecols=['timestamp', 'visitorid', 'itemid', 'click', 'pay',
                                                  'state', 'next_state'], nrows=self.nrows)  # timestamp, itemid
        len_items = len(self.features_dict)
        self.visitors = mdp_data.visitorid.unique().tolist()
        mdp_dataset = defaultdict(dict)
        pad = [0]*self.field_dims.shape[0]
        for i, d in mdp_data.groupby('visitorid'):
            d.sort_values(by='timestamp', inplace=True)
            labels = d[['click', 'pay']].values.astype(np.float32)
            s = [self.features_dict[j[0]].tolist() if j[0] in self.features_dict else pad for j in eval(d['next_state'].tolist()[-1])]
            cate_fea = np.array(s,dtype=np.int64)
            # padding numerical feature
            states = np.c_[cate_fea,np.zeros((cate_fea.shape[0],1))]
            # print(labels.shape,states.shape)
            mdp_dataset[i] = dict(
                labels=labels,
                states=states.astype(np.int64)
            )

        self.mdp_dataset = mdp_dataset
        self.datalen = len(self.visitors)
        print("visitors number:",len(self.visitors))

    def reset(self):
        visitorid = np.random.choice(self.visitors, size=1)[0]
        if self.is_test:
            visitorid = self.visitors[self.test_step%self.datalen]
            self.test_step += 1
        if self.is_seq:
            self.cur_session = seqVisitor(self.mdp_dataset[visitorid])
        else:
            self.cur_session = Visitor(self.mdp_dataset[visitorid],self.reward_type)
        return self.cur_session.state

    def step(self, action):  # offline behaviour, no need for action
        nstate, reward, done, label = self.cur_session.step(action)
        return nstate, reward, done, label

    def render(self):
        pass


if __name__ == '__main__':
    data_path = "./dataset/train.csv"
    features_path = "./dataset/rt/item_feadf.csv"
    map_path = "./chkpt"
    env = MTEnv(data_path, features_path, map_path, is_seq=False)
    env.getMDP()
    for i in range(10):
        state = env.reset()
        while True:
            action = env.action_space.sample()
            nstate, reward, done, _ = env.step(action)
            print("nstate:{},reward:{}".format(nstate,reward))
            if done:
                break

