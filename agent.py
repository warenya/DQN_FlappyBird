from imports import *

DATE_FORMAT = '%Y-%m-%d'
TIME_FORMAT = '%H:%M:%S'
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, hyperparam_set):
        with open('hyperparams.yaml', 'r') as f:
            all_hyperparams_set = yaml.safe_load(f)
            hyperparams = all_hyperparams_set[hyperparam_set]

        #Extract hyperparameters from hyperparams.yaml
        self.hyperparam_set = hyperparam_set
        self.env_id = hyperparams['env_id']
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.mini_batch_size = hyperparams['mini_batch_size']
        self.epsilon_init = hyperparams['epsilon_init']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
        self.network_sync_rate = hyperparams['network_sync_rate']
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.fc_layers = hyperparams['fc1_nodes']
        self.stop_on_reward = hyperparams['stop_on_reward']
        self.double_dqn = hyperparams['enable_double_dqn']
        self.dueling_dqn = hyperparams['enable_dueling_dqn']
        self.env_make_params = hyperparams.get('env_make_params', {})

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        #Files to store log, model and graph
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.png')

    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            global graph_update_time
            graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)} {start_time.strftime(TIME_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make( self.env_id,  render_mode="human" if render else None, **self.env_make_params)
        
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        dqn_policy = DQN(num_states, num_actions, self.fc_layers, self.dueling_dqn).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(capacity=self.replay_memory_size)
            target_model = DQN(num_states, num_actions, self.fc_layers, self.dueling_dqn).to(device)
            target_model.load_state_dict(dqn_policy.state_dict())
            self.optimizer = torch.optim.Adam(dqn_policy.parameters(), lr=self.learning_rate)
            epsilon_history = []
            step_count = 0
            best_reward = float('-inf')
        
        else:
            dqn_policy.load_state_dict(torch.load(self.MODEL_FILE))
            dqn_policy.eval()

        for episode in itertools.count():
            state,_ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            done = False
            episode_reward = 0.0

            while not done and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)
                else:
                    with torch.no_grad():
                        action = dqn_policy(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, done, _, info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float).to(device)
                reward = torch.tensor(reward, dtype=torch.float).to(device)

                if is_training:
                    memory.append((state, action, new_state, reward, done))
                    step_count += 1
                    
                #epsilon = self.epsilon
                state = new_state
                
            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = (f"{datetime.now().strftime(DATE_FORMAT)} \
                                   {datetime.now().strftime(TIME_FORMAT)}:New best reward {episode_reward:0.1f} \
                                   ({(episode_reward-best_reward)/best_reward*100:+.1f}%) \
                                   at episode {episode}, saving model...")
                    print(log_message)

                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')

                    torch.save(dqn_policy.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if (best_reward == self.stop_on_reward):
                    return 0

            rewards_per_episode.append(episode_reward)

            current_time = datetime.now()
            if current_time - graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                graph_update_time = current_time

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, dqn_policy, target_model)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

            if step_count > self.network_sync_rate:
                target_model.load_state_dict(dqn_policy.state_dict())
                step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))

        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, dqn_policy, target_model):
        states, actions, new_states, rewards, dones = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.double_dqn:
                best_action_from_policy = dqn_policy(new_states).argmax(dim=1)
                target_q = rewards + (1 - dones) * self.discount_factor * \
                           target_model(new_states).gather(dim=1, index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1 - dones) * self.discount_factor * \
                        target_model(new_states).max(dim=1)[0]
        current_q = dqn_policy(states).gather(1, actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--test', help='Testing mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparam_set=args.hyperparameters)

    # if args.train:
    #      dql.run(is_training=True)
    if args.test:
        dql.run(is_training=False, render=True)