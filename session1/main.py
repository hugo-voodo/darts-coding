import rooms
import agent as a
import matplotlib.pyplot as plot

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state)
        state = next_state
        discounted_return += reward*(discount_factor**time_step)
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return
    
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
nr_actions = env.action_space.n
discount_factor = 0.99
agent = a.PlanningAndLearningAgent(nr_actions, env, discount_factor, horizon=5, simulations=100, warmup_phase=5000)
training_episodes = 50
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

env.save_video()
