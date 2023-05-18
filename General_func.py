import sys
sys.path.append('/home/cedra/aseman rafat/paper code/time_series_augmentation')
sys.path.append('time_series_augmentation')
import numpy as np



def quat_multiply(quaternion0, quaternion1):
    x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
    x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate(
        (x1*w0 + y1*z0 - z1*y0 + w1*x0,
         -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
         -x1*x0 - y1*y0 - z1*z0 + w1*w0),
        axis=-1)


y_main = np.load('y.npy')
x_main = np.load('x.npy')

for nthrun in range(10):
    accuracies_rnn = []
    accuracies_cnn = []
    accuracies_best = []
    for ll in range(0, 20, 5):
        pass

def our_data_selection(x_main,y_main,N_dat,ll):
    y_train = []
    x_train = []
    tobedel = []
    for i in range(16):
        for j in range(ll, ll + N_dat):
            ii = 10 + i * 35 + j
            y_train.append(y_main[ii])
            x_train.append(x_main[ii])
            tobedel.append(ii)

    y = y_main.tolist()
    x = x_main.tolist()
    for index in sorted(tobedel, reverse=True):
        del y[index]
        del x[index]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x = np.array(x)
    y = np.array(y)
    return x_train,y_train,x,y


def get_stokoe(x):
    x_test_quad = x[:, :, [0, 2, 4, 6, 8, 10, 12]]
    x_test_fing1 = x[:, :, [14, 16, 18, 20, 22]]
    x_test_fing2 = x[:, :, list(range(1, 24, 2))]
    return x_test_quad,x_test_fing1,x_test_fing2



