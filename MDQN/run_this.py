from smac.env import StarCraft2Env
from RL_brain import DeepQNetwork
import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer

def run_this(RL, n_episode, learn_freq, Num_Exploration, n_agents, buffer_size, batch_size, gamma):
    step = 0
    training_step = 0
    n_actions_no_attack = 6
    replay_buffer = ReplayBuffer(buffer_size)
    for episode in range(n_episode):
        # initial observation
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0 for n in range(n_agents)]
        reward_hl_own_old = []
        reward_hl_en_old = []
        local_obs = env.get_obs()
        local_obs = np.array(local_obs)
        RL_input = np.expand_dims(local_obs, axis=0)
        for agent_id in range(n_agents):                #第一个循环是为了得到初始状态/观察/生命值信息
            reward_hl_own_old.append(env.get_agent_health(agent_id))
            reward_hl_en_old.append(env.get_enemy_health(agent_id))


        while True:
            # RL choose action based on local observation
            action_set_actual = []
            action_set_execute = []
            dead_unit = []
            RL_input = np.expand_dims(local_obs, axis=0)
            action_to_choose = RL.choose_action(RL_input)
            action_set_actual = action_to_choose

            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if action_set_actual[agent_id] in avail_actions_ind:
                    action_set_execute.append(action_set_actual[agent_id])
                elif(avail_actions[0] == 1):
                    action_set_execute.append(0)      #如果该动作不能执行，并且智能体已经死亡，那么就用NO_OP代替当前动作
                else:
                    action_set_execute.append(1)      #如果该动作不能执行，那么就用STOP动作代替

                if (len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):   #判断该智能体是否已经死亡
                    dead_unit.append(agent_id)

            # RL take action and get next observation and reward
            reward_base, done, _ = env.step(action_set_execute)
            episode_reward_all += reward_base
            new_local_obs = env.get_obs()
            new_local_obs = np.array(new_local_obs)
            reward_hl_own_new = []
            reward_hl_en_new = []

            for agent_id in range(n_agents):
                reward_hl_own_new.append(env.get_agent_health(agent_id))
                reward_hl_en_new.append(env.get_enemy_health(agent_id))

            rew_expand = np.zeros((n_agents, 1))
            # obtain propre reward of every agent and stored it in transition
            for agent_id in range(n_agents):
                if (agent_id in dead_unit):
                    reward = 0
                elif(action_set_execute[agent_id] != action_set_actual[agent_id]):  #当输出动作无法执行时，执行替代动作，但是把输出动作进行保存并且给与一个负的奖励
                    reward = -2

                elif(action_set_execute[agent_id] > 5):
                    target_id = action_set_execute[agent_id] - n_actions_no_attack
                    health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                    if(health_reduce_en > 0):
                        if(reward_base > 0):
                            reward = 2 + reward_base
                        else:
                            reward = 2
                    else:
                        reward = 1
                else:
                    reward = (reward_hl_own_new[agent_id] - reward_hl_own_old[agent_id]) * 5
                rew_expand[agent_id] = reward

                episode_reward_agent[agent_id] += reward

                replay_buffer.add(local_obs, action_set_actual, rew_expand, new_local_obs)

            # swap observation
            local_obs = new_local_obs
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                for i in range(n_agents):
                    RL.get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward_all))
                break

            step += 1

            if (step == Num_Exploration):
                print("Training starts.")

            if (step > Num_Exploration) and (step % learn_freq == 0):
                local_s_batch, a_batch, r_batch, local_s2_batch = replay_buffer.sample_batch(batch_size)  # [group0:[batch_size, trace.dimension], group1, ... group8]
                r_batch = np.squeeze(r_batch)
                eval_act_index = a_batch.astype(int)
                q_eval = RL.predict(local_s_batch)
                q_next = RL.predict_target(local_s2_batch)
                q_target = q_eval.copy()
                for i in range(batch_size):
                    for j in range(n_agents):
                        q_target[i, j, eval_act_index[i,j]] = r_batch[i,j] + gamma * np.max(q_next[i,j])
                RL.learn(local_s_batch, q_target)

                training_step += 1

            if (training_step >= 10000 and training_step % 10000 == 0):
                print("Model have been trained for %s times" % (training_step))


    # end of game
    print('game over')
    env.close()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="8m", reward_only_positive=False, obs_last_action=True, obs_timestep_number=True,
                        reward_scale_rate=200)  # 8m    reward_scale_rate=200
    env_info = env.get_env_info()

    vector_obs_len = 179  # local observation 80
    n_actions = env_info["n_actions"]
    n_episode = 2500   #每个episode大概能跑200步
    n_agents = env_info["n_agents"]
    # episode_len = env_info["episode_limit"]
    learn_freq = 1
    timesteps = 500000
    Num_Exploration = int(timesteps * 0.1)
    Num_Training = timesteps - Num_Exploration
    batch_size = 32
    g = tf.Graph()
    sess = tf.Session(graph=g)
    gamma = 0.99

    with sess.as_default():
        with g.as_default():

            RL = DeepQNetwork(n_actions=n_actions,
                              n_features=vector_obs_len,
                              n_agents=n_agents,
                              sess=sess,
                              num_training=Num_Training,
                              learning_rate=0.00025,   #0.002
                              reward_decay=0.99,
                              replace_target_iter=5000,
                              memory_size=Num_Exploration,
                              batch_size=32,
                              save_model_freq=20000,
                              load_model=False,
                              )

    # run_this写成一个所有智能体执行的函数
    run_this(RL, n_episode, learn_freq, Num_Exploration, n_agents, Num_Exploration, batch_size, gamma)