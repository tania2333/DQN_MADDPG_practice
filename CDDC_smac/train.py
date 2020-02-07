from smac.env import StarCraft2Env
from RL_brain import *
from replay_buffer import ReplayBuffer


def run_this(RL_set, n_episode, learn_freq, Num_Exploration, n_agents, n_actions, vector_obs_len, gamma, save_model_freq, batchsize, buffer_size):
    step = 0
    training_step = 0
    # n_actions_no_attack = 6
    action_list = []
    replay_buffer = ReplayBuffer(buffer_size)
    for n in range(n_actions):
        action_list.append(n)

    for episode in range(n_episode):
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

        while True:
            step = step + 1
            # critic_input = np.expand_dims(global_state_expand, axis=0)
            actor_input = np.expand_dims(local_obs, axis=0)
            action = RL_set[0][0].predict(actor_input)[0]
            act_with_noise = action  # np.clip(action + action_noise.get_noise(step_train), action_low, action_high)
            act_mat_norm = (act_with_noise + 1) / 2
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

                if (sum_avail_act == 0):
                    act_prob = (np.array(act_prob) + 1) / len(act_prob)
                else:
                    act_prob = np.array(act_prob) / sum_avail_act

                index = np.random.choice(np.array(avail_actions_ind), p=act_prob.ravel())
                actions.append(index)

                if (len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):
                    dead_unit.append(agent_id)

            reward_base, terminated, info = env.step(actions)

            new_local_obs = env.get_obs()
            new_local_obs = np.array(new_local_obs)
            new_global_state = env.get_state()
            new_global_state_expand = np.zeros(
                [new_local_obs.shape[0], new_local_obs.shape[1] + new_global_state.shape[0]])
            reward_hl_own_new = []
            reward_hl_en_new = []
            for i in range(new_local_obs.shape[0]):
                new_global_state_expand[i] = np.append(new_local_obs[i], new_global_state.flatten())
                reward_hl_own_new.append(env.get_agent_health(i))
                reward_hl_en_new.append(env.get_enemy_health(i))

            for i in range(n_agents):
                if (i in dead_unit):
                    rew_expand[i] = 0
                else:
                    rew_expand[i] = -0.05
                    if (actions[i] > 5):
                        target_id = actions[i] - 6
                        health_reduce_en = reward_hl_en_old[target_id] - reward_hl_en_new[target_id]
                        if (health_reduce_en > 0):
                            rew_expand[i] += 2 + health_reduce_en * 5
                        else:
                            rew_expand[i] += 1
                    else:
                        rew_expand[i] += (reward_hl_own_new[i] - reward_hl_own_old[i]) * 5
                #
                if (terminated):
                    if (info["battle_won"] is False):
                        rew_expand[i] += -10
                    else:
                        rew_expand[i] += 10

                episode_reward_agent[i] += rew_expand[i]

            replay_buffer.add(local_obs, global_state_expand, act_with_noise, rew_expand, terminated, new_local_obs,
                                  new_global_state_expand)
            # swap observation
            episode_reward += reward_base
            local_obs = new_local_obs
            global_state_expand = new_global_state_expand

            # break while loop when end of this episode
            if terminated:
                RL_set[0][1].get_episode_reward(episode_reward_agent[0], episode_reward, episode)
                for i in range(1, n_agents):
                    RL_set[i][0].get_episode_reward(episode_reward_agent[i], episode_reward, episode)
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward))
                break

            if (step == Num_Exploration):
                print("Training starts.")

            if (step > Num_Exploration) and (step % learn_freq == 0):
                training_step += 1
                grads_list = []
                actor = RL_set[0][0]
                local_s_batch, global_s_batch, a_batch, r_batch, done_batch, local_s2_batch, global_s2_batch = replay_buffer.sample_batch(
                    batch_size)
                actions_target_batch = actor.predict_target(local_s2_batch) #(batchsize, n_agents, n_actions)
                act_batch_input = actor.predict(local_s_batch)
                for agent_id in range(n_agents):
                    obs_batch = local_s_batch[:, agent_id, :]
                    next_obs_batch = local_s2_batch[:, agent_id, :]
                    act_batch = a_batch[:, agent_id, :]
                    rew_batch = r_batch[:, agent_id, :]

                    next_other_act_batch = np.delete(actions_target_batch, agent_id, 1) #(batch_size, n_agents - 1, n_actions) 可能会训练的不好，因为堆叠的时候 1 () 3 4 5 6,而maddpg中堆叠是(), 3，4，5，6，1
                    next_other_action = []
                    for i in range(n_agents-1):
                        next_other_action.append(next_other_act_batch[:, i, :])
                    next_other_action = np.hstack(next_other_action)

                    other_action = []
                    other_act_batch = np.delete(a_batch, agent_id, 1)
                    for i in range(n_agents-1):
                        other_action.append(other_act_batch[:, i, :])
                    other_action = np.hstack(other_action)  #((batch_size, n_agents - 1 * n_actions))

                    if(agent_id == 0):
                        critic = RL_set[0][1]
                    else:
                        critic = RL_set[agent_id][0]
                    #r_batch要对每个agent进行拆分
                    target_q = rew_batch.reshape(-1, 1) + gamma * critic.predict_target(next_obs_batch, actions_target_batch[:, agent_id, :], next_other_action)
                    predicted_q_value, critic_cost, _ = critic.train(obs_batch, act_batch, other_action, target_q)
                    critic.get_critic_loss(critic_cost)
                    act_batch_predict = act_batch_input[:, agent_id, :]
                    out, own_grads, other_grads = critic.action_gradients(obs_batch, act_batch_predict, other_action)  # delta Q对a的导数
                    own_grads = np.expand_dims(own_grads, axis=1)  #(32,1,14)
                    other_grads = other_grads.reshape(batchsize, n_agents-1, n_actions) #(32,7,14)
                    grads = np.concatenate((own_grads,other_grads), axis=1) #(32,8,14)
                    grads_list.append(grads)
                    critic.update_target_network()
                    # if (training_step % save_model_freq == 0):
                    #     critic.save_model(training_step)

                grads_array = np.array(grads_list)  #(8,32,8,14)
                actor.train(local_s_batch, grads_array)
                actor.update_target_network()

                if(training_step % save_model_freq == 0):
                    actor.save_model(training_step)

            if (training_step >= 10000 and training_step % 10000 == 0):
                print("Model have been trained for %s times" % (training_step))


    # end of game
    print('game over')
    env.close()


if __name__ == "__main__":
    env = StarCraft2Env(map_name="8m", reward_only_positive=False, obs_last_action=True, obs_timestep_number=True,
                        reward_scale_rate=200)  # 8m    reward_scale_rate=200
    env_info = env.get_env_info()
    local_obs_len = 179  # local obs：80 ; global state:168;
    global_state_len = 348  # 179+169
    hidden_vector_len = 256
    n_features = local_obs_len
    n_actions = env_info["n_actions"]
    n_episode = 2500 #3500  #每个episode大概能跑200步
    n_agents = env_info["n_agents"]
    # episode_len = env_info["episode_limit"]
    learn_freq = 1
    timesteps = 500000
    Num_Exploration = 70000 #int(timesteps * 0.1)  # 随着试验次数随时更改
    save_model_freq = 10000                              # 随着试验次数随时更改
    Num_Training = timesteps - Num_Exploration
    lr = 0.002
    tau = 0.001
    batch_size = 32
    actor_output_len = n_actions
    critic_output_len = 1
    reward_decay = 0.99  # 0.99
    # update_target_freq = 100
    load_model = False
    model_load_steps = 10000
    buffer_size = 70000


    agent_set = []

    for i in range(n_agents):
        g = tf.Graph()
        sess = tf.Session(graph=g)

        with sess.as_default():
            with g.as_default():
                net_set = []
                if(i == 0):
                    actor = ActorNetwork(sess, lr, tau, batch_size, n_agents, local_obs_len, actor_output_len, hidden_vector_len)
                    net_set.append(actor)
                critic = CriticNetwork(sess, lr, tau, n_agents, n_features, critic_output_len, n_actions, i)
                if (load_model):
                    actor.load_model(model_load_steps)
                else:
                    sess.run(tf.global_variables_initializer())

                net_set.append(critic)

        agent_set.append(net_set)

    # run_this写成一个所有智能体执行的函数
    run_this(agent_set, n_episode, learn_freq, Num_Exploration, n_agents, n_actions, local_obs_len, reward_decay, save_model_freq, batch_size, buffer_size)