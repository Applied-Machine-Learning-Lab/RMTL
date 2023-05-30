from env import MTEnv
from train.run import *
from agents.DDPG_ESMM_BC import TD3_ESMMBCAgent
from train.utils import ActionNormalizer
from train.Arguments import Arguments

def create_sub_agent(env, actor_name, agentcls, hyparams):
    hyparams.memory_size = 500000
    hyparams.init_episode = 200000
    hyparams.memory_path = "./pretrain/memory.pkl"
    hyparams.pretrain_path = f'chkpt/SL/rt_{actor_name}'
    hyparams.init_training_step = 1000
    hyparams.actor_reg = 0
    hyparams.critic_lr = 1e-3
    hyparams.ips = False

    agent = agentcls(env, actor_name, hyparams)
    hyparams.set_curpath(str(agent) + "_" + actor_name)

    return agent


if __name__ == '__main__':
    hyparams = Arguments()

    # 1. RL environment （采样视数据集数量选择行数，已有offline memory可将train_rows调到极低）
    hyparams.train_rows = 500
    env = ActionNormalizer(MTEnv(hyparams.train_path, hyparams.features_path, hyparams.map_path,
                                 reward_type=hyparams.reward_type, nrows=hyparams.train_rows))
    env.getMDP()

    # 2. Agent design
    agent = create_sub_agent(env, 'ple', TD3_ESMMBCAgent, hyparams)  # TD3BC agent, 0 as ac loss by default

    # 3. offline training
    hyparams.epoch = 96
    train_and_test_offline(env, agent, hyparams.epoch, hyparams.init_episode,
                           hyparams.save_dir, hyparams.memory_path)
