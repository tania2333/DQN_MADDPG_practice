from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
from BicNet.train_bicnet import *
# from smac.BicNet.train_comm_net import *

from BicNet.replay_buffer import ReplayBuffer
from BicNet.noise_OU import OU_noise
from baselines.logger import Logger, TensorBoardOutputFormat

from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf
import os
import datetime
import numpy as np
np.set_printoptions(threshold = 1e6)


def screenConcat(screen, num_agents):
  screen = screenReshape(screen)
  screen_final = screen
  if num_agents > 1:
    for i in range(num_agents-1):
      screen_final = np.concatenate((screen_final,screen),axis=0)
  return  screen_final

def screenReshape(screen):
  screen = np.array(screen)
  if (screen.shape[0] != 1):
    screen = screen.reshape((1, screen.shape[0] * screen.shape[1]))
  return screen

def state_transform(state_list):
  screen_final = np.array([])
  for i in range(len(state_list)):
    screen = state_list[i]
    screen = np.array(screen)
    screen = screen.reshape((1, screen.shape[0]))
    if(screen_final.size!=0):
      screen_final = np.concatenate((screen_final, screen), axis=0)
    else:
      screen_final = screen
  return  screen_final

def state_expand(state,n_agents):
    screen_final = np.array([])
    screen = state.reshape((1, state.shape[0]))
    for i in range(n_agents):
        if (screen_final.size != 0):
            screen_final = np.concatenate((screen_final, screen), axis=0)
        else:
            screen_final = screen.copy()
    return screen_final


def checkAction(act_value):
    if(act_value>=-1 and act_value<=-0.6):
        new_action = 1
    elif(act_value<=-0.2 and act_value>-0.6):
        new_action = 2
    elif(act_value<=0.2 and act_value>-0.2):
        new_action = 3
    elif (act_value <= 0.6 and act_value > 0.2):
        new_action = 4
    elif (act_value <= 1 and act_value > 0.6):
        new_action = 5
    else:
        new_action = 0
    return new_action

def single_list(arr, target):
    return arr.count(target)

def main():

    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    lr = 0.002
    buffer_size = 80000 #80000 # 减少一下，尽量是训练步数的1/10  70000  test 200  80000 20000
    batch_size = 32  # 32
    gamma = 0.99
    num_agents = 8
    local_obs_len = 179  # local obs：80 ; global state:168;

    output_len = 14
    hidden_vector_len = 256 #128  # 1  256
    tau = 0.001
    num_exploring = buffer_size #buffer_size
    action_low = -1
    action_high = 1
    save_freq = 10000
    # min_life = 45

    env = StarCraft2Env(map_name="8m", reward_only_positive=False, reward_scale_rate=200, obs_last_action=True, obs_timestep_number=True)  #8m  DefeatZerglingsAndBanelings  reward_scale_rate=200
    env_info = env.get_env_info()

    n_episodes = 4000 #4000    #2000
    # n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    episode_len = env_info["episode_limit"]

    timesteps = n_episodes * episode_len

    logdir = "tensorboard/zergling/%s/%s_num%s_lr%s/%s" % (
        "BicNet",
        timesteps,
        16,
        lr,
        start_time
    )

    Logger.DEFAULT \
        = Logger.CURRENT \
        = Logger(dir=None,
                 output_formats=[TensorBoardOutputFormat(logdir)])

    sess = U.make_session()
    sess.__enter__()
    # state_dim = (n_agents, vector_obs_len)
    # action_dim = (n_agents, output_len)

    actor = ActorNetwork(sess, lr, tau, batch_size, num_agents, vector_obs_len, output_len, hidden_vector_len)
    # actor = ActorNetwork(sess, state_dim, action_dim, lr, tau, batch_size)
    critic = CriticNetwork(sess, lr, tau, actor.get_num_trainable_vars(), num_agents, vector_obs_len,
                              output_len, hidden_vector_len)
    # critic = CriticNetwork(sess, state_dim, action_dim, lr, tau, gamma, actor.get_num_trainable_vars())
    sess.run(tf.global_variables_initializer())
    replay_buffer = ReplayBuffer(buffer_size)
    action_noise = OU_noise(decay_period=timesteps - buffer_size)

    action_noise.reset()
    # model_file_load = os.path.join(str(350000) + "_" + "model_segment_training2/", "defeat_zerglings")
    # U.load_state(model_file_load, sess)
    U.initialize()


    t = 0
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        obs = env.get_obs()
        obs = np.array(obs)
        # state, target_attack = env.get_state()
        state = env.get_state()
        screen_expand = np.zeros([obs.shape[0],obs.shape[1] + state.shape[0]])
        for i in range(obs.shape[0]):
            screen_expand[i] = np.append(obs[i],state.flatten())
        # screen_expand = state_transform(obs)
        # screen_expand = state_expand(state, n_agents)
        while not terminated:
            t = t+1
            screen_input = np.expand_dims(screen_expand, axis=0)
            action = actor.predict(screen_input)[0]
            act_with_noise = np.clip(action + action_noise.get_noise(t - num_exploring), action_low, action_high)
            act_mat_norm = (act_with_noise+1)/2
            actions = []
            dead_unit = []
            rew_expand = np.zeros((n_agents, 1))
            # punish = []
            # health_agent = []
            # health_enemy = []

            agent_group = []
            for agent_id in range(n_agents):
                sum_avail_act = 0
                act_prob = []
                obs_agent = env.get_obs_agent(agent_id)     # for test
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                act_unit_norm = act_mat_norm[agent_id]
                # print('act_unit_norm',act_unit_norm)
                # act_prob = act_unit_norm / np.sum(act_unit_norm, axis=0)
                for i in avail_actions_ind:
                    act_prob.append(act_unit_norm[i])
                    sum_avail_act = sum_avail_act + act_unit_norm[i]

                if(sum_avail_act == 0):
                    act_prob = (np.array(act_prob) + 1)/len(act_prob)
                else : act_prob = np.array(act_prob)/sum_avail_act

                # index = np.random.choice(np.arange(0,14), p=act_prob.ravel())
                # print("act_prob",act_prob)
                index = np.random.choice(np.array(avail_actions_ind), p=act_prob.ravel())
                # if (index in avail_actions_ind):
                #     punish.append(False)
                # else:
                #     punish.append(True)
                #     if (0 in avail_actions_ind):
                #         actions.append(0)
                #     else:
                #         actions.append(1)
                actions.append(index)
                # health_agent.append(state[4*agent_id])
                # health_enemy.append(state[4*n_agents + 3*agent_id])

                # if(index > 5):
                #     target_id = index - 6

                if(len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):
                    dead_unit.append(agent_id)
            # health_agent = np.array(health_agent)
            # for i in range(len(health_enemy)):
            #     if (health_enemy[i] < min_life):
            #         min_life = health_enemy[i]
            # health_enemy = np.array(health_enemy)
            reward, terminated, _ = env.step(actions)
            # rew_expand = np.ones((n_agents, 1))*reward
            # health_enemy_new = []


            for i in range(n_agents):
                if (i not in dead_unit):
                    rew_expand[i] += reward
                    if (actions[i] > 5):
                        enemy_id = actions[i] - 6
                        rew_expand[i] += 1
                        # if(actions[i]-6 == target_attack):
                        # for j in range(n_agents):
                            # if (actions[j] == actions[i] and i!=j):
                                # if (state[4 * n_agents + 3 * enemy_id] == min):
                                #     rew_expand[i] += 1
            new_obs = env.get_obs()
            new_obs = np.array(new_obs)
            # new_state, target_attack = env.get_state()
            new_state= env.get_state()
            new_screen_expand = np.zeros([new_obs.shape[0], new_obs.shape[1] + new_state.shape[0]])
            for i in range(new_obs.shape[0]):
                new_screen_expand[i] = np.append(new_obs[i], new_state.flatten())
            # health_agent_new = []
            # for i in range(n_agents):
            #     health_agent_new.append(new_state[4 * i])
            #     # health_enemy_new.append(new_state[4 * n_agents + 3 * i])
            # health_agent_new = np.array(health_agent_new)
            # health_enemy_new = np.array(health_enemy_new)
            # life_reduce_agent = health_agent - health_agent_new
            # life_reduce_agent_all = life_reduce_agent.sum(axis=0)
            # life_reduce_enemy = health_enemy - health_enemy_new
            # life_reduce_enemy_all = life_reduce_enemy.sum(axis=0)
            # reward_base = life_reduce_enemy_all - life_reduce_agent_all
            # for i in range(n_agents):
            #     rew_expand[i] += reward_base+life_reduce_agent[i]

            # for i in range(n_agents):
            #     if (punish[i]):
            #         rew_expand[i] += -2
            #     elif (i in dead_unit):
            #         rew_expand[i] += 0
            #     elif (actions[i] > 5):
            #         rew_expand[i] = 1
            #         if(health_enemy[actions[i] - 6] == min_life):
            #             rew_expand[i] = 1
            #     rew_expand[i] += life_reduce_agent[i]

            replay_buffer.add(screen_expand, act_with_noise, rew_expand, terminated, new_screen_expand)

            episode_reward += reward
            screen_expand = new_screen_expand
            # state = new_state
            # target_attack = target_attack_new

            if(t>=num_exploring):
                print("training starts")
                s_batch, a_batch, r_batch, done_batch, s2_batch = replay_buffer.sample_batch(batch_size)  # [group0:[batch_size, trace.dimension], group1, ... group8]
                target_q = r_batch + gamma * critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(target_q, (batch_size, num_agents, output_len)))
                a_outs = actor.predict(s_batch)  # a_outs和a_batch是完全相同的
                grads = critic.action_gradients(s_batch, a_outs)  # delta Q对a的导数
                actor.train(s_batch, grads)

                actor.update_target_network()
                critic.update_target_network()

                # if(t % save_freq == 0):
                    # model_file_save = os.path.join(str(t) + "_" + "model_segment_training/", "defeat_zerglings")
                    # U.save_state(model_file_save)
                    # replay_buffer.save()

        print("Total reward in episode {} = {}".format(e, episode_reward))
        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", e)
        logger.record_tabular("reward", episode_reward)

        logger.dump_tabular()

    # model_file_save = os.path.join(str(t) + "_" + "model_segment_training/", "defeat_zerglings")
    # U.save_state(model_file_save)

    env.close()


if __name__ == "__main__":
    main()
