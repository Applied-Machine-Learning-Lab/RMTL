import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .DDPG_ESMM import DDPG_wESMMAgent
import warnings

warnings.filterwarnings("ignore")


class TD3_ESMMBCAgent(DDPG_wESMMAgent):
    def __init__(self, env: gym.Env, actor_name, arguments):
        super_args = dict(
            env=env,
            actor_name=actor_name,
            embed_dim=arguments.embed_dim,
            ou_noise_theta=arguments.ou_noise_theta,
            ou_noise_gamma=arguments.ou_noise_gamma,
            gamma=arguments.gamma,
            memory_size=arguments.memory_size,
            batch_size=arguments.batch_size,
            drop_out=arguments.drop_out,
            pretrain_path=arguments.pretrain_path,
            actor_lr=arguments.actor_lr,
            critic_lr=arguments.critic_lr,
            actor_reg=arguments.actor_reg,
            tau=arguments.tau,
            soft_update_freq=arguments.soft_update_freq,
            actor_update_freq=arguments.actor_update_freq,
            init_training_step=arguments.init_training_step,
            ips=arguments.ips,
        )
        super(TD3_ESMMBCAgent, self).__init__(**super_args)
        self.actor_target_optimizer = optim.Adam(self.actor_target.parameters(), lr=5*self.actor_lr)

    def state_normalize(self, state):
        mu = state.mean(axis=0)
        std = state.std(axis=0)
        return (state-mu)/(std+1e-3)

    def process_batch(self, transition):
        state = transition['state']
        nstate = transition['nstate']
        # TD3BC approach 1: no need for cate features
        state, nstate = self.state_normalize(state), self.state_normalize(nstate)
        # end

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

        # TD3 next action sample:
        noise = torch.clip(self.noise.sample(), -3e-3, 3e-3).to(self.device).detach()
        naction1, naction2 = naction1+noise[0], naction2+noise[1]
        # end

        res = dict(
            state=(cate_features, num_features),
            action=(action1, action2),
            reward=(reward1, reward2),
            nstate=(ncate_features, nnum_features),
            naction=(naction1, naction2),
            mask=mask,
            label=(label[:,0], label[:, 1])
        )
        return res

    def get_closs(self, critic_id, critic, critic_target, transition):
        q_pred = critic(transition["state"][0], transition["state"][1], transition["action"][critic_id])
        # TD3 approach: 取小target
        q_target = torch.min(
            torch.stack(
                [critic_target(transition["nstate"][0], transition["nstate"][1], transition["naction"][critic_id]),
                 critic(transition["nstate"][0], transition["nstate"][1], transition["naction"][critic_id])],
                dim=-1),
            dim=-1
        ).values
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
        a_loss = torch.mean(a_loss)

        if self.actor_reg > 0:
            ref_loss = 0
            param_count = 0
            for param, value in self.actor.named_parameters():
                param_count += 1
                ref_loss += F.mse_loss(value,
                                       self.pretain_actor.state_dict()[param])
            ref_loss /= param_count

            # TD3BC: approach 3
            # mask
            # a_loss = a_loss + self.actor_reg * ref_loss
            # add lines
            lambda_ = self.actor_reg / torch.mean(-q1_loss_weight - q2_loss_weight).abs().detach()
            a_loss = lambda_ * a_loss + ref_loss      # reg the true actor?
            # ac_loss = lambda_ * ac_loss + ref_loss  # no reg here over reference actor
            # self.actor_target_optimizer.zero_grad()
            # ac_loss.backward()
            # self.actor_target_optimizer.step()
            # end

        return a_loss, ac_loss

    def update(self):
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

            # 不更新a，只做记录
            a_loss, ac_loss = self.get_aloss(transition=transition)
            # update target networks
            self._target_soft_update(self.tau)

            critic_lossls1.append(q1_loss.item())
            critic_lossls2.append(q2_loss.item())
            actor_lossls1.append(ac_loss.item())
            actor_lossls2.append(a_loss.item())
            self.total_step += 1
        return np.mean(critic_lossls1), np.mean(critic_lossls2), np.mean(actor_lossls1), np.mean(actor_lossls2)


    def __str__(self):
        return "TD3BC"
