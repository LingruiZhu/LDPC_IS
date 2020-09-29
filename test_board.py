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

a = np.array([[1, 2, 3, 4 ,5],
              [10, 20, 30, 40, 50]])
print(a[1,:].shape)
b = a[1,:]
print(b.ndim)


a = [1, 1, 1, 1]
aa = [2, 2, 2, 2]
bb = [i/j for i,j in zip(a, aa)]
print(bb)