import matplotlib.pyplot as plt
import numpy as np
import sys

from environment import MountainCar

def get_state(mode, state):

    if mode == 'raw':
        s = np.zeros((2, 1))
    else:
        s = np.zeros((2048, 1))

    for k in state:
        s[k] = state[k]

    return s

def get_q(s, w, b):

    return w.T @ s + b

def train_q_learning(mode,
                     episodes,
                     max_episode_len,
                     epsilon,
                     gamma,
                     lr):

    n_S = 2048
    n_A = 3

    if mode == 'raw':
        n_S = 2

    w = np.zeros((n_S, n_A))
    b = 0

    returns = []

    car = MountainCar(mode)

    for i in range(episodes):

        s = get_state(mode, car.reset())
        R = 0.
        g = 1.

        for t in range(max_episode_len):

            assert s.shape[0] <= n_S
            isGreedy = np.random.uniform(0, 1, 1) >= epsilon

            if isGreedy:
                a = int(np.argmax(get_q(s, w, b)))
            else:
                a = int(np.random.randint(0, n_A, 1))

            state, r, done = car.step(a)
            s_ = get_state(mode, state)

            R += (r * g)
            # g *= gamma

            q_sa = float(get_q(s, w, b)[a])

            # if done:
            #     max_q_s_ = 0
            # else:
            max_q_s_ = float(np.max(get_q(s_, w, b)))

            td = q_sa - (r + gamma * max_q_s_)

            w[:, a] = w[:, a] - lr * td * s.flatten()
            b = b - lr * td

            s = s_

            if done:
               break

        returns.append(R)

    return np.array(returns), w, b

def write_array_to_file(filepath, array):
    """Write a numpy matrix to a file.

    Args:
        filepath (str): File path.
        array (ndarray): Numpy 1D array.
    """

    assert len(array.shape) < 2

    out_str = '\n'.join(array.astype('U'))

    with open(filepath, 'w') as f:
        f.write(out_str)

    print('Array written to {0} successfully!'.format(filepath))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    assert len(sys.argv) == 1 + 8

    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_episode_len = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    lr = float(sys.argv[8])

    print(f'{mode = }')
    print(f'{weight_out = }')
    print(f'{returns_out = }')
    print(f'{episodes = }')
    print(f'{max_episode_len = }')
    print(f'{epsilon = }')
    print(f'{gamma = }')
    print(f'{lr = }')

    returns, w, b = train_q_learning(mode,
                                     episodes,
                                     max_episode_len,
                                     epsilon,
                                     gamma,
                                     lr)

    weights = np.concatenate((np.array([b]), w.flatten()))

    write_array_to_file(weight_out, weights)
    write_array_to_file(returns_out, returns)

    plt.plot(list(range(1, episodes + 1)), returns, 'ro-', label='returns')
    plt.plot(list(range(25, episodes + 1)), moving_average(returns, 25), 'bo-', label='rolling_mean')
    plt.title('Tile')
    plt.legend()
    plt.show()