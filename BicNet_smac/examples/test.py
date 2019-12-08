from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
from smac.BicNet.train_bicnet import *
# from smac.BicNet.train_comm_net import *

from smac.BicNet.replay_buffer import ReplayBuffer
from smac.BicNet.noise_OU import OU_noise
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
    batch_size = 32  # 32
    num_agents = 8
    vector_obs_len = 248  # local obs：80 ; global state:168;
    output_len = 14
    hidden_vector_len = 256 #128  # 1  256
    tau = 0.001



    env = StarCraft2Env(map_name="8m",reward_only_positive=False, reward_scale_rate=200)  #8m  DefeatZerglingsAndBanelings  reward_scale_rate=200
    env_info = env.get_env_info()

    n_episodes = 4000 #4000    #2000
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

    actor = ActorNetwork(sess, lr, tau, batch_size, num_agents, vector_obs_len, output_len, hidden_vector_len)
    critic = CriticNetwork(sess, lr, tau, actor.get_num_trainable_vars(), num_agents, vector_obs_len,
                              output_len, hidden_vector_len)
    sess.run(tf.global_variables_initializer())
    model_file_load = os.path.join(str(80000) + "_" + "model_segment_training/", "defeat_zerglings")
    U.load_state(model_file_load, sess)

    t = 0
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        obs = env.get_obs()
        obs = np.array(obs)
        state, min = env.get_state()
        screen_expand = np.zeros([obs.shape[0],obs.shape[1] + state.shape[0]])
        for i in range(obs.shape[0]):
            screen_expand[i] = np.append(obs[i],state.flatten())

        while not terminated:
            t = t+1
            screen_input = np.expand_dims(screen_expand, axis=0)
            action = actor.predict(screen_input)[0]
            act_with_noise = np.clip(action, -1, 1)
            act_mat_norm = (act_with_noise+1)/2
            actions = []

            for agent_id in range(n_agents):
                sum_avail_act = 0
                act_prob = []
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                act_unit_norm = act_mat_norm[agent_id]
                for i in avail_actions_ind:
                    act_prob.append(act_unit_norm[i])
                    sum_avail_act = sum_avail_act + act_unit_norm[i]

                if(sum_avail_act == 0):
                    act_prob = (np.array(act_prob) + 1)/len(act_prob)
                else : act_prob = np.array(act_prob)/sum_avail_act
                index = np.random.choice(np.array(avail_actions_ind), p=act_prob.ravel())

                actions.append(index)
            reward, terminated, _ = env.step(actions)

            new_obs = env.get_obs()
            new_obs = np.array(new_obs)
            new_state, min = env.get_state()
            new_screen_expand = np.zeros([new_obs.shape[0], new_obs.shape[1] + new_state.shape[0]])
            for i in range(new_obs.shape[0]):
                new_screen_expand[i] = np.append(new_obs[i], new_state.flatten())

            episode_reward += reward
            screen_expand = new_screen_expand

        print("Total reward in episode {} = {}".format(e, episode_reward))
        logger.record_tabular("steps", t)
        logger.record_tabular("episodes", e)
        logger.record_tabular("reward", episode_reward)

        logger.dump_tabular()

    env.close()


if __name__ == "__main__":
    main()
