from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
from BicNet.train_bicnet import *

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

def main():

    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    env = StarCraft2Env(map_name="8m", reward_only_positive=False, reward_scale_rate=200, state_last_action=True,
                        obs_last_action=True, obs_timestep_number=True, state_timestep_number=True)
    env_info = env.get_env_info()

    n_episodes = 200 #4000    #2000
    # timesteps = 700000
    n_agents = env_info["n_agents"]
    n_actions= env_info["n_actions"]
    output_len = n_actions
    lr = 0.002
    # buffer_size = int(timesteps * 0.1 / 70 )  # 80000 # 减少一下，尽量是训练步数的1/10  70000  test 200  80000 20000
    batch_size = 32  # 32
    # gamma = 0.99
    num_agents = 8
    local_obs_len = 179  # local obs：80 ; global state:168;
    global_state_len = 348   # 179+169

    hidden_vector_len = 256  # 128  # 1  256
    tau = 0.001
    # num_exploring = buffer_size  # buffer_size
    action_low = -1
    action_high = 1
    # save_freq = 1000 #10000
    critic_output_len = n_actions

    # logdir = "tensorboard/%s/%s_lr%s/%s" % (
    #     "BicNet",
    #     timesteps,
    #     lr,
    #     start_time
    # )
    #
    # Logger.DEFAULT \
    #     = Logger.CURRENT \
    #     = Logger(dir=None,
    #              output_formats=[TensorBoardOutputFormat(logdir)])

    sess = U.make_session()
    sess.__enter__()

    actor = ActorNetwork(sess, lr, tau, batch_size, num_agents, local_obs_len, output_len, hidden_vector_len)
    critic = CriticNetwork(sess, lr, tau, actor.get_num_trainable_vars(), num_agents, global_state_len,
                              critic_output_len, hidden_vector_len, n_actions)
    sess.run(tf.global_variables_initializer())
    # replay_buffer = ReplayBuffer(buffer_size)
    # action_noise = OU_noise(decay_period=timesteps - buffer_size)

    # action_noise.reset()
    U.initialize()
    model_file_load = os.path.join("model/"+str(60001) + "_" + "training_steps_model/", "8m")
    U.load_state(model_file_load, sess)

    t = 0
    # step_train = 0
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        local_obs = env.get_obs()
        local_obs = np.array(local_obs)
        global_state = env.get_state()
        global_state_expand = np.zeros([local_obs.shape[0], local_obs.shape[1] + global_state.shape[0]])
        reward_hl_own_old = []
        reward_hl_en_old = []
        episode_reward_agent = [0 for n in range(n_agents)]
        for i in range(local_obs.shape[0]):
            global_state_expand[i] = np.append(local_obs[i], global_state.flatten())
            reward_hl_own_old.append(env.get_agent_health(i))
            reward_hl_en_old.append(env.get_enemy_health(i))


        while not terminated:
            t = t+1
            critic_input = np.expand_dims(global_state_expand, axis=0)
            actor_input = np.expand_dims(local_obs, axis=0)
            action = actor.predict(actor_input)[0]
            act_with_no_noise = np.clip(action, action_low, action_high)
            act_mat_norm = (act_with_no_noise+1)/2
            actions = []
            dead_unit = []
            rew_expand = np.zeros((n_agents, 1))

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

                if(len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):
                    dead_unit.append(agent_id)

            reward_base, terminated, _ = env.step(actions)

            new_local_obs = env.get_obs()
            new_local_obs = np.array(new_local_obs)
            new_global_state= env.get_state()
            new_global_state_expand = np.zeros([new_local_obs.shape[0], new_local_obs.shape[1] + new_global_state.shape[0]])
            reward_hl_own_new = []
            reward_hl_en_new = []
            for i in range(new_local_obs.shape[0]):
                new_global_state_expand[i] = np.append(new_local_obs[i], new_global_state.flatten())
                reward_hl_own_new.append(env.get_agent_health(i))
                reward_hl_en_new.append(env.get_enemy_health(i))

            for i in range(n_agents):
                if (i in dead_unit):
                    rew_expand[i] = 0
                elif (actions[i] > 5):
                    target_id = actions[i] - 6
                    health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    if (health_reduce_en > 0):
                        if (reward_base > 0):
                            rew_expand[i] = 2 + reward_base
                        else:
                            rew_expand[i] = 2
                    else:
                        rew_expand[i] = 1
                else:
                    rew_expand[i] = (reward_hl_own_new[i] - reward_hl_own_old[i]) * 5

                episode_reward_agent[i] += rew_expand[i]

            episode_reward += reward_base
            local_obs = new_local_obs
            global_state_expand = new_global_state_expand

        print("Total reward in episode {} = {}".format(e, episode_reward))
        # logger.record_tabular("steps", t)
        # logger.record_tabular("episodes", e)
        # logger.record_tabular("reward_episode", episode_reward)
        # for i in range(n_agents):
        #     logger.record_tabular("reward_agent_"+str(i), episode_reward_agent[i])
        #
        # logger.dump_tabular()

    # model_file_save = os.path.join(str(t) + "_" + "model_segment_training/", "defeat_zerglings")
    # U.save_state(model_file_save)
    print('game over')
    print(env.get_stats())
    env.close()


if __name__ == "__main__":
    main()
