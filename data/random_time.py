import numpy as np

# Produce test set
x = np.random.randint(70000, 110000, size=1000)
np.save('data/test_random_time_1000.npy', x)

# Produce valid set
x = np.random.randint(50000, 70000, size=150)
np.save(f'data/valid_random_time_150.npy', x)
