import argparse
import numpy as np
import tensorflow as tf
import pickle
from smac.env import StarCraft2Env
from baselines import logger
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="8m", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=300, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=50, help="number of episodes")
    parser.add_argument("--buffer-size", type=int, default=5000, help="maximum storage size of replay buffer")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=512, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="8m", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=25, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./model/model_4087027steps/8m", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=512, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(num_adversaries, obs_shape_n, action_space_n, arglist, buffer_size):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_space_n, i, arglist, buffer_size,
            local_q_func=(arglist.adv_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = StarCraft2Env(map_name=arglist.scenario, reward_only_positive=False, obs_last_action=True, obs_timestep_number=True,
                            reward_scale_rate=200)
        # Create agent trainers
        env_info = env.get_env_info()
        num_agents = env_info["n_agents"]
        num_adversaries = num_agents
        obs_shape_n = [(env_info["obs_shape"], ) for i in range(num_adversaries)]
        action_space_n = [env_info["n_actions"] for i in range(num_adversaries)]
        buffer_size = arglist.buffer_size

        trainers = get_trainers(num_adversaries, obs_shape_n, action_space_n, arglist, buffer_size)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # logdir = "./tensorboard/"
        #
        # Logger.DEFAULT \
        #     = Logger.CURRENT \
        #     = Logger(dir=None,
        #              output_formats=[TensorBoardOutputFormat(logdir)])

        # Load previous results, if necessary
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(num_agents)]  # individual agent reward
        saver = tf.train.Saver()
        n_actions_no_attack = 6

        env.reset()

        obs_n = []
        reward_hl_own_old = []
        reward_hl_en_old = []
        for agent_id in range(num_agents):  # 第一个循环是为了得到初始状态/观察/生命值信息
            obs = env.get_obs_agent(agent_id)
            obs_n.append(obs)
            reward_hl_own_old.append(env.get_agent_health(agent_id))
            reward_hl_en_old.append(env.get_enemy_health(agent_id))

        episode_step = 0
        step = 0

        print('Starting iterations...')
        while True:
            # get action
            action_set_actual = []
            action_set_execute = []
            action_n = []
            dead_unit = []
            for agent_id in range(num_agents):
                action_output = trainers[agent_id].action(obs_n[agent_id])
                action_n.append(action_output)
                action_prob = action_output
                action_to_choose = np.argmax(action_prob)
                action_set_actual.append(action_to_choose)
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if action_to_choose in avail_actions_ind:
                    action_set_execute.append(action_to_choose)
                elif (avail_actions[0] == 1):
                    action_set_execute.append(0)  # 如果该动作不能执行，并且智能体已经死亡，那么就用NO_OP代替当前动作
                else:
                    action_set_execute.append(1)  # 如果该动作不能执行，那么就用STOP动作代替

                if (len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):  # 判断该智能体是否已经死亡
                    dead_unit.append(agent_id)

            rew_base, done, _ = env.step(action_set_execute)
            episode_rewards[-1] += rew_base
            new_obs_n = []
            reward_hl_own_new = []
            reward_hl_en_new = []
            rew_n = []

            for agent_id in range(num_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                new_obs_n.append(obs_next)
                reward_hl_own_new.append(env.get_agent_health(agent_id))
                reward_hl_en_new.append(env.get_enemy_health(agent_id))

            for agent_id in range(num_agents):
                if (agent_id in dead_unit):
                    reward = 0
                elif(action_set_execute[agent_id] != action_set_actual[agent_id]):  #当输出动作无法执行时，执行替代动作，但是把输出动作进行保存并且给与一个负的奖励
                    reward = -2

                elif(action_set_execute[agent_id] > 5):
                    target_id = action_set_execute[agent_id] - n_actions_no_attack
                    health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    if(health_reduce_en > 0):
                        if(rew_base > 0):
                            reward = 2 + rew_base
                        else:
                            reward = 2
                    else:
                        reward = 1
                else:
                    reward = (reward_hl_own_new[agent_id] - reward_hl_own_old[agent_id]) * 5
                rew_n.append(reward)

            episode_step += 1

            # collect experience
            # for i, agent in enumerate(trainers):
            #     agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done)

            obs_n = new_obs_n
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            if done:
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, len(episode_rewards), episode_rewards[-1]))
                # logger.record_tabular("episodes", len(episode_rewards))
                # logger.record_tabular("episode reward", episode_rewards[-1])
                # for i in range(num_agents):
                #     logger.record_tabular("agent"+str(i)+" episode reward",
                #                           agent_rewards[i][-1])
                # logger.dump_tabular()

                env.reset()
                obs_n = []
                reward_hl_own_old = []
                reward_hl_en_old = []
                for agent_id in range(num_agents):  # 第一个循环是为了得到初始状态/观察/生命值信息
                    obs = env.get_obs_agent(agent_id)
                    obs_n.append(obs)
                    reward_hl_own_old.append(env.get_agent_health(agent_id))
                    reward_hl_en_old.append(env.get_enemy_health(agent_id))
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

            # increment global step counter
            step += 1
            # if(step == arglist.buffer_size):
            #     print("Training starts.")

            # update all trainers, if not in display or benchmark mode
            # loss = None
            # for agent in trainers:
            #     agent.preupdate()
            # for agent in trainers:
            #     loss = agent.update(trainers, step)

            # save model, display training output
            # if done and (len(episode_rewards) % arglist.save_rate == 0):
            #     save_dir = arglist.save_dir + "/model_" + str(step) +"steps/" + arglist.exp_name
            #     U.save_state(save_dir, saver=saver)
            #     # print statement depends on whether or not there are adversaries
            #     if num_adversaries == 0:
            #         print("steps: {}, episodes: {}, mean episode reward: {}".format(
            #             step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:])))
            #     else:
            #         print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
            #             step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
            #             [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards) - 1))
                print(env.get_stats())
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
