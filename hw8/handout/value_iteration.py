import numpy as np

v_t = np.zeros((2, 3))

rewards = np.array([[0, -100, -100, -100, -100],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, -100, -100, -100, -100]])

n = 3

for i in range(n):
    pad_vt = np.pad(v_t, 1)

    # print(rewards[1: -1, 1: -1].shape, pad_vt[1: -1, 1: -1].shape)
    # print(rewards[1: -1, : -2].shape, pad_vt[1: -1, : -2].shape)
    # print(rewards[1: -1, 2:].shape, pad_vt[1: -1, 2:].shape)

    v_t1 = 0.8 * (rewards[1: -1, 1: -1] + pad_vt[1: -1, 1: -1]) + \
        0.1 * (rewards[1: -1, : -2] + pad_vt[1: -1, : -2]) + \
        0.1 * (rewards[1: -1, 2:] + pad_vt[1: -1, 2:])

    v_t = v_t1

    print(v_t1)