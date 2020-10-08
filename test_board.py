import pandas as pd
import numpy as np

dic = {'[2, 2]': [], '[4, 1]': []}
print(dic)
ts_size = [3,1]
ts_v_bits = [12, 33, 61]

if str(ts_size) in dic.keys():
    dic[str(ts_size)].append(ts_v_bits)
else:
    dic[str(ts_size)] = []
    dic[str(ts_size)].append(ts_v_bits)

print(dic)

ts_size = [2,2]
ts_v_bits = [18, 13]

if str(ts_size) in dic.keys():
    dic[str(ts_size)].append(ts_v_bits)
else:
    dic[str(ts_size)] = []
    dic[str(ts_size)].append(ts_v_bits)

print(dic)

ts_size = [2,2]
ts_v_bits = [18, 13]

if str(ts_size) in dic.keys():
    if ts_v_bits not in dic[str(ts_size)]:
        dic[str(ts_size)].append(ts_v_bits)
else:
    dic[str(ts_size)] = []
    dic[str(ts_size)].append(ts_v_bits)

print(dic)

print('dictionary saved')
np.save('test_dict.npy', dic)

print('dictionary read')
dic_read = np.load('test_dict.npy', allow_pickle='True').item()
print(dic_read)

t = [0, 1, 2, 3]
t.append(4)
print(t)
print(t[0])

print(len(dic.keys()))

ts_file = 'trapping_set_folder/100loops/ts_SNR_5_dB_100loops.npy'
ts_dict = np.load(ts_file, allow_pickle=True).item()
num_ts = 0
for key in ts_dict.keys():
    num_ts = num_ts + len(ts_dict[key])
print(num_ts)

a = np.array([[1, 1, 1],
              [1, 1, 1]])

b = np.array([[2, 2, 2],
              [2, 2, 2]])

print(a/b)

c = np.array([[0, 0, 0, 0]])
d = np.array([[1, 2, 3, 4]])
e = np.array([[5, 6, 7, 8]])

c = np.concatenate([c,d], axis=0)
print(c)
print(np.mean(c))

a_list = [4, 2]
a_tuple = tuple(a_list)
print(a_tuple)
print(a_tuple[0])
print(a_tuple[1])

# test different methods of calculating the stand deviation
import numpy.random as rnd
from datetime import datetime

now = datetime.now()
rng = rnd.default_rng(now.microsecond)

mu = 0
sigma = 2
n_samples = 1000
a = rng.normal(0, 2, [n_samples, 1])
a2 = a.reshape([10, 100])
a2_sup = np.mean(a2, axis=0)
print(a2_sup.shape)

std1 = np.std(a)
mean1 = np.mean(a)
print(mean1, std1)

std2 = np.std(a2_sup)
mean2 = np.mean(a2_sup)
print(mean2, std2)

import math
from scipy import stats
import matplotlib.pyplot as plt


mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 300)

mu_biased = 2
sigma_biased = 1.75*sigma

plt.subplot(121)
plt.title('Mean Translation')
plt.grid()
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='original density f', color='red')
plt.plot(x, stats.norm.pdf(x, mu_biased, sigma), label='biased density g', color='blue')
plt.plot(x, stats.norm.pdf(x, mu_biased, sigma))
plt.legend()

plt.subplot(122)
plt.title('Variance Scaling')
plt.grid()
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='original density f', color='red')
plt.plot(x, stats.norm.pdf(x, mu, sigma_biased), label='biased density g', color='blue')
plt.legend()

plt.show()
