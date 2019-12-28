from smac.env import StarCraft2Env
from RL_brain2 import *

def run_this(RL_set, n_episode, learn_freq, Num_Exploration, n_agents, n_actions, vector_obs_len, gamma, save_model_freq, batchsize):
    step = 0
    training_step = 0
    n_actions_no_attack = 6
    action_list = []
    for n in range(n_actions):
        action_list.append(n)

    for episode in range(n_episode):
        # initial observation
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0 for n in range(n_agents)]
        observation_set = []
        reward_hl_own_old = []
        reward_hl_en_old = []
        for agent_id in range(n_agents):                #第一个循环是为了得到初始状态/观察/生命值信息
            obs = env.get_obs_agent(agent_id)
            obs = obs.reshape((1, vector_obs_len))
            observation_set.append(obs)
            reward_hl_own_old.append(env.get_agent_health(agent_id))
            reward_hl_en_old.append(env.get_enemy_health(agent_id))

        while True:
            # RL choose action based on local observation
            action_set_actual = []
            action_set_execute = []
            action_output_set = []
            dead_unit = []
            for agent_id in range(n_agents):
                action_output = RL_set[agent_id][0].predict(observation_set[agent_id])
                action_output_set.append(action_output)
                action_prob = action_output
                action_to_choose = np.argmax(action_prob)
                action_set_actual.append(action_to_choose)
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if action_to_choose in avail_actions_ind:
                    action_set_execute.append(action_to_choose)
                elif(avail_actions[0] == 1):
                    action_set_execute.append(0)      #如果该动作不能执行，并且智能体已经死亡，那么就用NO_OP代替当前动作
                else:
                    action_set_execute.append(1)      #如果该动作不能执行，那么就用STOP动作代替

                if (len(avail_actions_ind) == 1 and avail_actions_ind[0] == 0):   #判断该智能体是否已经死亡
                    dead_unit.append(agent_id)

            # RL take action and get next observation and reward
            reward_base, done, _ = env.step(action_set_execute)
            episode_reward_all += reward_base
            observation_set_next = []
            reward_hl_own_new = []
            reward_hl_en_new = []

            for agent_id in range(n_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                obs_next = obs_next.reshape((1, vector_obs_len))
                observation_set_next.append(obs_next)
                reward_hl_own_new.append(env.get_agent_health(agent_id))
                reward_hl_en_new.append(env.get_enemy_health(agent_id))

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

                episode_reward_agent[agent_id] += reward
                RL_set[agent_id][0].store_transition(observation_set, action_output_set, reward, observation_set_next, done)  #action_set_actual

            # swap observation
            observation_set = observation_set_next
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                for i in range(n_agents):
                    RL_set[i][1].get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward_all))
                break

            step += 1

            if (step == Num_Exploration):
                print("Training starts.")

            if (step > Num_Exploration) and (step % learn_freq == 0):
                for agent_id in range(n_agents):
                    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = RL_set[agent_id][0].memory.sample(batchsize)
                    act_batch = total_act_batch[:, 0, :]  # 0 代表当前智能体的动作

                    other_act_batch = []
                    for i in range(1, n_agents):
                        other_act_batch.append(total_act_batch[:, i, :])
                    other_act_batch = np.hstack(other_act_batch)  # 其他智能体的动作，可以将其放到critic网络进行训练
                    obs_batch = total_obs_batch[:, 0, :]  # 当前智能体局部观察
                    next_obs_batch = total_next_obs_batch[:, 0, :]

                    other_actors = []
                    if (agent_id != n_agents - 1):
                        for i in range(agent_id + 1, n_agents):
                            other_actors.append(RL_set[i])
                    if (agent_id != 0):
                        for i in range(agent_id):
                            other_actors.append(RL_set[i])
                    next_other_action = []
                    for i in range(1, n_agents):
                        next_other_action.append(other_actors[i - 1][0].predict_target(total_next_obs_batch[:, i, :]))
                    next_other_action = np.hstack(next_other_action)
                    actor = RL_set[agent_id][0]
                    critic = RL_set[agent_id][1]

                    target_q = rew_batch.reshape(-1, 1) + gamma * critic.predict_target(next_obs_batch, actor.predict_target(next_obs_batch), next_other_action)
                    # orig_param_critic = critic.get_params()
                    predicted_q_value, critic_cost, _ = critic.train(obs_batch, act_batch, other_act_batch, target_q)
                    # new_param_critic = critic.get_params()
                    RL_set[agent_id][1].get_critic_loss(critic_cost)
                    act_batch_input = actor.predict(obs_batch)
                    out, grads = critic.action_gradients(obs_batch, act_batch_input, other_act_batch)  # delta Q对a的导数
                    actor.train(obs_batch, grads)
                    # if(training_step % update_target_net == 0):
                    actor.update_target_network()
                    critic.update_target_network()

                    if(training_step % save_model_freq == 0):
                        actor.save_model(training_step)
                        # critic.save_model(training_step)


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
    n_features = vector_obs_len
    n_actions = env_info["n_actions"]
    n_episode = 3500   #每个episode大概能跑200步
    n_agents = env_info["n_agents"]
    # episode_len = env_info["episode_limit"]
    learn_freq = 1
    timesteps = 700000
    Num_Exploration = int(timesteps * 0.1)         # 随着试验次数随时更改
    save_model_freq = 20000                              # 随着试验次数随时更改
    Num_Training = timesteps - Num_Exploration
    learning_rate_actor = 1e-4
    learning_rate_critic = 1e-3
    tau = 0.01
    batch_size = 64
    output_len = 1
    reward_decay = 0.99  # 0.99
    # update_target_freq = 100
    load_model = False
    model_load_steps = 20000

    agent_set = []

    for i in range(n_agents):
        g = tf.Graph()
        sess = tf.Session(graph=g)

        with sess.as_default():
            with g.as_default():
                net_set = []
                actor = ActorNetwork(sess, learning_rate_actor, tau, batch_size, n_agents, n_features, n_actions, i, memory_size=Num_Training,
                                     num_training=Num_Training, test_flag=False)
                critic = CriticNetwork(sess, learning_rate_critic, tau, n_agents, n_features, output_len, n_actions, i)
                if (load_model):
                    actor.load_model(model_load_steps)
                    # critic.load_model(model_load_steps)
                else:
                    sess.run(tf.global_variables_initializer())

                net_set.append(actor)
                net_set.append(critic)

        agent_set.append(net_set)

    # run_this写成一个所有智能体执行的函数
    run_this(agent_set, n_episode, learn_freq, Num_Exploration, n_agents, n_actions, vector_obs_len, reward_decay, save_model_freq, batch_size)