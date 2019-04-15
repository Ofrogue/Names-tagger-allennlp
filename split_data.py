import os
import random

data = open(os.path.join('data', 'output.txt')).read().split('\n')


random.shuffle(data)

train_size = int(len(data) * 0.6)


f = open(os.path.join('data', 'train.txt'), 'w')
f.write('\n'.join(data[:train_size]))
f.close()

f = open(os.path.join('data', 'val.txt'), 'w')
f.write('\n'.join(data[train_size:]))
f.close()