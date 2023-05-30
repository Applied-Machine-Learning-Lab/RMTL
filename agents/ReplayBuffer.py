import numpy as np
from typing import Dict, List, Tuple
class ReplayBuffer:
    def __init__(self,state_dim:int, action_dim:int, size:int, batch_size:int):
        self.state_buf = np.zeros([size,state_dim],dtype=np.float32)
        self.action_buf = np.zeros([size,action_dim],dtype=np.float32)
        self.label_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.nstate_buf = np.zeros([size,state_dim],dtype=np.float32)
        # self.reward_buf = np.zeros([size],dtype=np.float32)
        self.reward_buf = np.zeros([size,2], dtype=np.float32)
        self.done_buf = np.zeros([size],dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.idx, self.size = 0,0

    def store(self,state:np.ndarray,action:np.ndarray,nstate:np.ndarray,reward:float,done:bool,label:float):
        self.state_buf[self.idx] = state
        self.nstate_buf[self.idx] = nstate
        self.action_buf[self.idx] = action
        self.label_buf[self.idx] = label
        self.reward_buf[self.idx] = reward
        self.done_buf[self.idx] = done
        self.size = min(1+self.size, self.max_size)
        self.idx = (1+self.idx)%self.max_size

    def sample_batch(self)->Dict[str,np.ndarray]:  # 类型有点问题
        choices = np.random.choice(self.size, size=self.batch_size,replace=False)  # 注意用法
        return dict(
            state = self.state_buf[choices],
            action = self.action_buf[choices],
            nstate = self.nstate_buf[choices],
            reward = self.reward_buf[choices],
            done = self.done_buf[choices],
            label = self.label_buf[choices]
            ) # 对象型数据结构

    def __len__(self)->int:
        return self.size