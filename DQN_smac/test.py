from smac.env import StarCraft2Env
from RL_brain import DeepQNetwork
import datetime
import numpy as np
import tensorflow as tf

def run_this(RL_set, n_episode, n_agents):
    step = 0
    for episode in range(n_episode):
        # initial observation
        env.reset()
        episode_reward_all = 0
        episode_reward_agent = [0 for n in range(n_agents)]
        observation_set = []
        reward_hl_own_old = []
        reward_hl_en_old = []
        for agent_id in range(n_agents):
            obs = env.get_obs_agent(agent_id)
            observation_set.append(obs)
            reward_hl_own_old.append(obs[-1])
            reward_hl_en_old.append(env.get_enemy_health(agent_id))

        while True:
            # RL choose action based on observation
            action_set_actual = []
            action_set_execute = []
            for agent_id in range(n_agents):
                action_to_choose = RL_set[agent_id].choose_action(observation_set[agent_id])
                action_set_actual.append(action_to_choose)
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if action_to_choose in avail_actions_ind:
                    action_set_execute.append(action_to_choose)
                elif(avail_actions[0] == 1):
                    action_set_execute.append(0)      #如果该动作不能执行，并且智能体已经死亡，那么就用NO_OP代替当前动作
                else:
                    action_set_execute.append(1)      #如果该动作不能执行，那么就用STOP动作代替

            # RL take action and get next observation and reward
            reward_base, done, _ = env.step(action_set_execute)
            episode_reward_all += reward_base
            observation_set_next = []
            reward_hl_own_new = []
            reward_hl_en_new = []

            for agent_id in range(n_agents):
                obs_next = env.get_obs_agent(agent_id=agent_id)
                observation_set_next.append(obs_next)
                reward_hl_own_new.append(obs_next[-1])
                reward_hl_en_new.append(env.get_enemy_health(agent_id))

                if (action_set_execute[agent_id] > 5):
                    reward = reward_base + (reward_hl_en_old[agent_id] - reward_hl_en_new[agent_id]) - (
                                reward_hl_own_old[agent_id] - reward_hl_own_new[agent_id])
                else:
                    reward = reward_base - (reward_hl_own_old[agent_id] - reward_hl_own_new[agent_id])

                episode_reward_agent[agent_id] += reward

            # swap observation
            observation_set = observation_set_next
            reward_hl_own_old = reward_hl_own_new
            reward_hl_en_old = reward_hl_en_new

            # break while loop when end of this episode
            if done:
                print("steps until now : %s, episode: %s" % (step, episode))
                break

            step += 1


    # end of game
    print('game over')
    env.close()


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    env = StarCraft2Env(map_name="8m", reward_only_positive=False,
                        reward_scale_rate=200)  # 8m    reward_scale_rate=200
    env_info = env.get_env_info()

    vector_obs_len = 80  # local observation
    n_actions = env_info["n_actions"]
    n_episode = 40
    n_agents = env_info["n_agents"]
    episode_len = env_info["episode_limit"]
    # timesteps = 800000
    # learn_freq = 1

    RL_set = []
    graph_set = []
    sess_set = []
    for i in range(n_agents):
        g = tf.Graph()
        sess = tf.Session(graph=g)

        with sess.as_default():
            with g.as_default():

                RL = DeepQNetwork(n_actions=n_actions,
                                  n_features=vector_obs_len,
                                  sess=sess,
                                  agent_id=i,
                                  learning_rate=0.002,
                                  reward_decay=0.99,
                                  replace_target_iter=5000,
                                  memory_size=80000,
                                  batch_size=32,
                                  save_model_freq=10000,
                                  load_model=True,
                                  )

                RL_set.append(RL)

    # run_this写成一个所有智能体执行的函数
    run_this(RL_set, n_episode, n_agents)