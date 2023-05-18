# This is a sample Python script.
from General_func import *
from DMP_aug import *
from deep_models import *




def main():
    y_main = np.load('y.npy')
    x_main = np.load('x.npy')
    N_dat = 5
    ll = 5
    x_train, y_train, x, y = our_data_selection(x_main,y_main,N_dat,ll)
    x_test_quad, x_test_fing1, x_test_fing2 = get_stokoe(x)
    x_train_quad, x_train_fing1, x_train_fing2 = get_stokoe(x_train)
    dmp_list_quad = quaternion_dmp_maker(x_train_quad, duration = 5.0)
    dmp_list_fing1 = fing1_dmp_maker(x_train_fing1, duration = 5.0)
    x_test = test_array(x_test_quad,x_test_fing1,x_test_fing2,duration=5)
    x_train, y_train = aug_generator(20, dmp_list_fing1, dmp_list_quad, x_train_fing2, y_train)
    LSTM_model(x_test, y, x_train, y_train)
    CNN_model(x_test, y, x_train, y_train)

main()


