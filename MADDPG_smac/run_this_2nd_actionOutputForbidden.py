from smac.env import StarCraft2Env
from RL_brain import MADDPG
import numpy as np
import tensorflow as tf


class OU_noise(object):
	def __init__(self, n_actions, action_low, action_high, decay_period, mu=0.0, theta=0.1, max_sigma=0.2, min_sigma=0.05):
		self.mu = mu
		self.theta = theta
		self.sigma = max_sigma
		self.max_sigma = max_sigma
		self.min_sigma = min_sigma
		self.decay_period = decay_period
		self.num_actions = n_actions
		self.action_low = action_low
		self.action_high = action_high
		self.reset()

	def reset(self):
		self.state = np.zeros(self.num_actions)

	def state_update(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)
		self.state = x + dx

	def add_noise(self, action, training_step):
		self.state_update()
		state = self.state
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, training_step / self.decay_period)
		return np.clip(action + state, self.action_low, self.action_high)


def run_this(RL_set, n_episode, learn_freq, Num_Exploration, Num_Training, n_agents, n_actions, vector_obs_len):
    step = 0
    training_step = 0
    n_actions_no_attack = 6
    action_list = []
    noise = OU_noise(n_actions, -1, 1, decay_period=Num_Training)
    for n in range(n_actions):
        action_list.append(n)

    for episode in range(n_episode):
        # initial observation
        env.reset()
        noise.reset()
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
                action_output = noise.add_noise(RL_set[agent_id].action(observation_set[agent_id]), training_step)
                action_output_set.append(action_output)
                action_norm = (action_output + 1) / 2
                action_sum = action_norm.sum()
                action_prob = action_norm / action_sum
                action_to_choose = np.random.choice(action_list, p=action_prob.ravel())
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

            reward_set = []
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
                reward_set.append(reward)

            for agent_id in range(n_agents):
                RL_set[agent_id].store_transition(observation_set, action_output_set, reward_set[agent_id], observation_set_next, done)  #action_set_actual

            # swap observation
            observation_set = observation_set_next
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                for i in range(n_agents):
                    RL_set[i].get_episode_reward(episode_reward_agent[i], episode_reward_all, episode)
                print("steps until now : %s, episode: %s， episode reward: %s" % (step, episode, episode_reward_all))
                break

            step += 1

            if (step == Num_Exploration):
                print("Training starts.")

            if (step > Num_Exploration) and (step % learn_freq == 0):
                for agent_id in range(n_agents):
                    RL_set[agent_id].train_agent(RL_set)
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
    n_episode = 4000   #每个episode大概能跑200步
    n_agents = env_info["n_agents"]
    # episode_len = env_info["episode_limit"]
    learn_freq = 1
    timesteps = 800000
    Num_Exploration = int(timesteps * 0.1)
    Num_Training = timesteps - Num_Exploration

    agent_set = []

    for i in range(n_agents):
        g = tf.Graph()
        sess = tf.Session(graph=g)

        with sess.as_default():
            with g.as_default():

                agent = MADDPG(
                    nb_other_aciton=98,
                    n_actions=n_actions,
                    n_agents=8,
                    n_features=vector_obs_len,
                    sess=sess,
                    agent_id=i,
                    num_training=Num_Training,
                    reward_decay=0.99,
                    replace_target_iter=5000,
                    memory_size=Num_Exploration,
                    batch_size=64,
                    save_model_freq=20000,
                    load_model=False,
                    )
                agent_set.append(agent)
    # run_this写成一个所有智能体执行的函数
    run_this(agent_set, n_episode, learn_freq, Num_Exploration, Num_Training, n_agents, n_actions, vector_obs_len)