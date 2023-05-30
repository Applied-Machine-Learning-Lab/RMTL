import os
from types import MethodType
from typing import Tuple
from copy import deepcopy
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from .ReplayBuffer import ReplayBuffer
from layers.critic import Critic,CriticNeg
from train.run import get_model
from train.utils import get_optim_param
import warnings

warnings.filterwarnings("ignore")


class DDPG_wESMMAgent:
    def __init__(self,
                 env: gym.Env,
                 actor_name="esmm",
                 embed_dim=128,
                 bottom_mlp_dims=(512, 256),
                 tower_mlp_dims=(128, 64),
                 ou_noise_theta=0.1,
                 ou_noise_gamma=0.4,
                 gamma=0.9,
                 memory_size=100000,
                 batch_size=512,
                 drop_out=0.2,
                 actor_lr=0,
                 critic_lr=1e-3,
                 actor_reg=3e-2,
                 tau=0.2,
                 soft_update_freq=2,
                 actor_update_freq=2,
                 init_training_step=10000,
                 ips=True,
                 pretrain_path="../pretrain",
                 ):
        # system parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.gamma = gamma
        action_dim = self.env.action_space.shape[0]
        cate_dim = self.env.field_dims

        # training param
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.pretrain_path = pretrain_path
        self.memory = ReplayBuffer(cate_dim.shape[0] + 1, action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.tau = tau
        self.actor_reg = actor_reg
        self.soft_update_freq = soft_update_freq
        self.actor_update_freq = actor_update_freq
        self.init_training_step = init_training_step
        self.ips = ips

        # actor param
        self.categorical_field_dims = cate_dim
        self.num_dim = 1
        self.embed_dim = embed_dim
        self.bottom_mlp_dims = bottom_mlp_dims
        self.tower_mlp_dims = tower_mlp_dims
        self.task_num = action_dim
        self.drop_out = drop_out
        critic_model = CriticNeg

        # define actor network
        self.pretain_actor = get_model(actor_name,self.categorical_field_dims, self.num_dim, self.task_num, 8,
                                self.embed_dim).to(self.device)

        self.critic1 = critic_model(self.categorical_field_dims, self.num_dim, self.embed_dim, self.bottom_mlp_dims,
                              self.tower_mlp_dims, self.drop_out).to(self.device)

        self.critic2 = deepcopy(self.critic1)

        self.pretain_actor.load_state_dict(torch.load(self.pretrain_path+f"/rt_{actor_name}_0.0.pt"))
        self.pretain_actor.eval()
        self.actor = deepcopy(self.pretain_actor)
        self.critic1.embedding.load_state_dict(self.actor.embedding.state_dict())
        self.critic2.embedding.load_state_dict(self.critic1.embedding.state_dict())
        if os.path.exists(self.pretrain_path + "/critic1.pth") and os.path.exists(self.pretrain_path + "/critic2.pth"):
            state_dict1 = torch.load(self.pretrain_path + "/critic1.pth", map_location=lambda storage, loc: storage)
            self.critic1.load_state_dict(state_dict1)
            state_dict2 = torch.load(self.pretrain_path + "/critic2.pth", map_location=lambda storage, loc: storage)
            self.critic2.load_state_dict(state_dict2)

        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # self.aloss_reg = torch.autograd.Variable(torch.FloatTensor([1.]).to(self.device), requires_grad=True)

        self.critic1_target = deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)

        self.critic2_target = deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

        self.actor_optimizer.parameters = MethodType(get_optim_param, self.actor_optimizer)
        self.critic1_optimizer.parameters = MethodType(get_optim_param, self.critic1_optimizer)
        self.critic2_optimizer.parameters = MethodType(get_optim_param, self.critic2_optimizer)

        self.noise = Normal(ou_noise_theta * torch.ones(self.task_num), ou_noise_gamma * torch.ones(self.task_num))

        self.transition = dict()

        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if len(state.shape) == 1:  # deal with 1-d dimension
            state = np.expand_dims(state, 0)
        cate_features, num_features = torch.LongTensor(state[:, :-1]).to(self.device), \
                                      torch.FloatTensor(state[:, [-1]]).to(self.device)
        # print("features:",cate_features)
        selected_action = torch.stack(self.actor(cate_features, num_features), 1)
        # print(selected_action)
        selected_action = selected_action.cpu().detach().numpy()

        if not self.is_test:
            # attention: explore
            noise = np.clip(self.noise.sample().cpu().detach().numpy(), -3e-3, 3e-3)
            selected_action = selected_action + noise
        return selected_action.reshape(-1)

    def process_batch(self, transition):
        state = transition['state']
        nstate = transition['nstate']
        cate_features, num_features = torch.LongTensor(state[:, :-1]).to(self.device), \
                                      torch.FloatTensor(state[:, [-1]]).to(self.device)
        ncate_features, nnum_features = torch.LongTensor(nstate[:, :-1]).to(self.device), \
                                        torch.FloatTensor(nstate[:, [-1]]).to(self.device)

        action = torch.FloatTensor(transition['action']).to(self.device)
        reward = torch.FloatTensor(transition['reward'].reshape(-1, 2)).to(self.device)
        mask = torch.FloatTensor(1 - transition['done'].reshape(-1)).to(self.device)
        label = torch.FloatTensor(transition['label']).to(self.device)
        naction = torch.stack(self.actor_target(ncate_features, nnum_features), 1)
        reward1 = reward[:, 0]
        reward2 = reward[:, 1]
        action1, action2 = torch.unsqueeze(action[:, 0], 1), torch.unsqueeze(action[:, 1], 1)
        naction1, naction2 = torch.unsqueeze(naction[:, 0], 1), torch.unsqueeze(naction[:, 1], 1)

        res = dict(
            state=(cate_features, num_features),
            action=(action1, action2),
            reward=(reward1, reward2),
            nstate=(ncate_features, nnum_features),
            naction=(naction1, naction2),
            mask=mask,
            label=(label[:, 0], label[:, 1])
        )
        return res

    def get_closs(self, critic_id, critic, critic_target, transition):
        q_pred = critic(transition["state"][0], transition["state"][1], transition["action"][critic_id])
        q_target = critic_target(transition["nstate"][0], transition["nstate"][1], transition["naction"][critic_id])
        q_target = transition["reward"][critic_id] + self.gamma * q_target * transition["mask"]
        q_loss = torch.mean(
            torch.multiply(F.mse_loss(q_pred, q_target.detach(), reduce=False), transition["weight"][critic_id]))
        return q_loss

    def get_aloss(self, transition):
        ref_action = self.actor(transition["state"][0], transition["state"][1])
        # seprate AC loss by q
        q1_loss_weight = -torch.multiply(
            self.critic1(transition["state"][0], transition["state"][1],
                         torch.unsqueeze(ref_action[0], 1)),
            transition["weight"][0])
        q2_loss_weight = - torch.multiply(
            self.critic2(transition["state"][0], transition["state"][1],
                         torch.unsqueeze(ref_action[1], 1)),
            transition["weight"][1])
        ac_loss = torch.mean(q1_loss_weight + q2_loss_weight)  # refer to normal ac loss
        a_loss = ac_loss

        if self.actor_reg > 0:
            ref_loss = 0
            param_count = 0
            for param, value in self.actor.named_parameters():
                param_count += 1
                ref_loss += F.mse_loss(value,
                                       self.pretain_actor.get_parameter(param))
            ref_loss /= param_count
            a_loss += self.actor_reg * ref_loss
        return a_loss, ac_loss

    def update(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        update_steps = len(self.memory) // self.batch_size
        actor_lossls1 = []
        actor_lossls2 = []
        critic_lossls1 = []
        critic_lossls2 = []
        for i in range(update_steps):
            tb = self.memory.sample_batch()
            transition = self.process_batch(tb)
            # IPS weight
            if self.ips:
                label = torch.stack(transition["label"], dim=-1)
                pos = (self.batch_size + 1) / (torch.sum(label, dim=0) + torch.ones(2).to(self.device))
                w_pos = torch.unsqueeze(pos, dim=0).repeat(self.batch_size, 1)
                w_neg = (self.batch_size + 1) / (self.batch_size - torch.sum(label, dim=0))
                weight = label * w_pos + w_neg
            else:
                weight = torch.ones((self.batch_size, 2)).to(self.device)

            transition["weight"] = (weight[:, 0], weight[:, 1])

            # update critic
            q1_loss = self.get_closs(critic_id=0, critic=self.critic1, critic_target=self.critic1_target,
                                     transition=transition)
            self.critic1_optimizer.zero_grad()
            q1_loss.backward()
            self.critic1_optimizer.step()

            q2_loss = self.get_closs(critic_id=1, critic=self.critic2, critic_target=self.critic2_target,
                                     transition=transition)
            self.critic2_optimizer.zero_grad()
            q2_loss.backward()
            self.critic2_optimizer.step()

            # update actor: nabla_{\theta}\pi_{\theta}nabla_{a}Q_{\pi}(s,a)
            a_loss, ac_loss = self.get_aloss(transition=transition)

            if (self.total_step + 1) % self.actor_update_freq == 0 and self.total_step > self.init_training_step:
                self.actor_optimizer.zero_grad()
                a_loss.backward()
                self.actor_optimizer.step()

                # update target networks
                self._target_soft_update(self.tau)

            critic_lossls1.append(q1_loss.item())
            critic_lossls2.append(q2_loss.item())
            actor_lossls1.append(ac_loss.item())
            actor_lossls2.append(a_loss.item())
            self.total_step += 1
        return np.mean(critic_lossls1), np.mean(critic_lossls2), np.mean(actor_lossls1), np.mean(actor_lossls2)

    def _target_soft_update(self, tau: float):
        if self.total_step % self.soft_update_freq == 0:
            for t_param, l_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

            for t_param, l_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

            for t_param, l_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.actor), ('act_optim', self.actor_optimizer),
                         ('critic1', self.critic1), ('cri1_optim', self.critic1_optimizer),
                         ('critic2', self.critic2), ('cri2_optim', self.critic2_optimizer)]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def __str__(self):
        return "DDPG_ESMM"
