import sys
sys.path.append('/home/cedra/aseman rafat/paper code/time_series_augmentation')
sys.path.append('time_series_augmentation')
from movement_primitives.dmp import DMP, CartesianDMP
import numpy as np
import utils.augmentation as aug


def multi_dimension_dmp(my_array, duration = 5.0):
    t = np.linspace(0, duration, my_array.shape[0])
    mydmp = DMP(my_array.shape[1], duration, n_weights_per_dim= 6, dt=0.104)
    mydmp.imitate(t, my_array, regularization_coefficient=0, allow_final_velocity = False)
    return mydmp


def quaternion_dmp_maker(x_train_quad, duration = 5.0):
    dmp_list_quat = []
    for one_train_quad in x_train_quad:
        t = np.linspace(0, duration, len(one_train_quad))
        mydmp = CartesianDMP(2, n_weights_per_dim=6, dt=0.041)
        mydmp.imitate(t, one_train_quad)
        # (T,yyy3) = mydmp.open_loop()
        # dmp_dict_quad.append(yyy3)
        dmp_list_quat.append(mydmp)
    return dmp_list_quat


def fing1_dmp_maker(x_train_fing1, duration = 5.0):
    dmp_list_fing1 = []
    for one_train_fing1 in x_train_fing1:
        t = np.linspace(0, duration, len(one_train_fing1))
        mydmp = DMP(5, 2, n_weights_per_dim=6, dt=0.041)
        mydmp.imitate(t, one_train_fing1)
        # (T,yyy3) = mydmp.open_loop()
        # dmp_dict_quad.append(yyy3)
        dmp_list_fing1.append(mydmp)
    return dmp_list_fing1


def test_array(x_test_quad,x_test_fing1,x_test_fing2,duration=5):
    x_test = []
    for i in range(x_test_quad.shape[0]):
        one_test_quad = x_test_quad[i]
        one_test_fing1 = x_test_fing1[i]
        one_test_fing2 = x_test_fing2[i]

        t = np.linspace(0, duration, len(one_test_quad))
        mydmp1 = CartesianDMP(2, n_weights_per_dim=6, dt=0.041)
        mydmp1.imitate(t, one_test_quad, allow_final_velocity=True)
        (tt, yyy1) = mydmp1.open_loop()

        mydmp2 = DMP(5, 2, n_weights_per_dim=6, dt=0.041)
        mydmp2.imitate(t, one_test_fing1, allow_final_velocity=True)
        (tt, yyy2) = mydmp2.open_loop()

        yyy3 = x_test_fing2[i]
        yyy = np.concatenate((yyy1, yyy2, yyy3), axis=1)
        x_test.append(yyy)

    x_test = np.array(x_test)
    return x_test



def dmp_gen_fing1(dmp_dict_fing1, alpha_z, alpha_y):
    dmp_dict_fing11 = []
    for thedmp1 in dmp_dict_fing1:
        (T,ymain) = thedmp1.open_loop()
        diff_y1 = (np.max(ymain,axis=0)-np.min(ymain,axis=0))
        startmain1 = thedmp1.start_y
        goalmain1 = thedmp1.goal_y

        thedmp1.forcing_term.alpha_z = np.random.choice(alpha_z)
        thedmp1.alpha_y = np.random.choice(alpha_y)
        thedmp1.beta_y = thedmp1.alpha_y/4

        sampl1 = np.random.uniform(low=-diff_y1[:5]*0.5, high=diff_y1[:5]*0.5, size=(5,))
        sampl11 = np.random.uniform(low=-diff_y1*0.5, high=diff_y1*0.5, size=(5,))
        sampl12 = np.random.uniform(low=-diff_y1*0.5, high=diff_y1*0.5, size=(5,))

        thedmp1.start_y = thedmp1.start_y + sampl11
        thedmp1.goal_y = thedmp1.goal_y + sampl12

        (T,yy1) = thedmp1.open_loop()
        yy1[:,:5] = yy1[:,:5] + sampl1

        thedmp1.alpha_y = 25
        thedmp1.beta_y = 6.25
        thedmp1.forcing_term.alpha_z = 4.59987
        thedmp1.start_y = startmain1
        thedmp1.goal_y = goalmain1
        dmp_dict_fing11.append(yy1)

        return dmp_dict_fing11


def dmp_gen_quat(dmp_dict_quad, alpha_z, alpha_y):
    dmp_dict_quad2 = []
    for thedmp1 in dmp_dict_quad:
        (T,ymain) = thedmp1.open_loop()
        diff_y1 = (np.max(ymain,axis=0)-np.min(ymain,axis=0))
        startmain1 = thedmp1.start_y
        goalmain1 = thedmp1.goal_y

        thedmp1.forcing_term_pos.alpha_z = np.random.choice(alpha_z)
        thedmp1.alpha_y = np.random.choice(alpha_y)
        thedmp1.forcing_term_rot.alpha_z = np.random.choice(alpha_z)
        thedmp1.beta_y = thedmp1.alpha_y/4

        sampl1 = np.random.uniform(low=-diff_y1[:3]*0.3, high=diff_y1[:3]*0.3, size=(3,))
        sampl11 = np.random.uniform(low=-diff_y1*0.3, high=diff_y1*0.3, size=(7,))
        sampl12 = np.random.uniform(low=-diff_y1*0.3, high=diff_y1*0.3, size=(7,))

        thedmp1.start_y = thedmp1.start_y + sampl11
        thedmp1.goal_y = thedmp1.goal_y + sampl12

        thedmp1.start_y[3:] = thedmp1.start_y[3:]/np.sqrt(thedmp1.start_y[3]**2+thedmp1.start_y[4]**2+thedmp1.start_y[5]**2+thedmp1.start_y[6]**2)
        thedmp1.goal_y[3:] = thedmp1.goal_y[3:]/np.sqrt(thedmp1.goal_y[3]**2+thedmp1.goal_y[4]**2+thedmp1.start_y[5]**2+thedmp1.goal_y[6]**2)

        (T,yy1) = thedmp1.open_loop()
        yy1[:,:3] = yy1[:,:3] + sampl1

        thedmp1.alpha_y = 25
        thedmp1.beta_y = 6.25
        thedmp1.forcing_term_rot.alpha_z = 4.59987
        thedmp1.forcing_term_pos.alpha_z = 4.59987
        thedmp1.start_y = startmain1
        thedmp1.goal_y = goalmain1
        dmp_dict_quad2.append(yy1)

        return dmp_dict_quad2


def aug_generator(num_rep, dmp_dict_fing1, dmp_dict_quad, x_train_fing2, y_train):
    new_y = []
    new_x = []
    alpha_z = np.random.normal(4.6, 0.5, 2000)
    alpha_y = np.random.normal(25, 5, 2000)
    for i in range(num_rep):
        dmp_dict_fing11 = (dmp_dict_fing1)
        dmp_dict_quad2 = dmp_gen_quat(dmp_dict_quad)
        augmentdata1 = aug.magnitude_warp(aug.time_warp(aug.permutation(x_train_fing2)))
        augmentdata = np.concatenate((dmp_dict_quad2, dmp_dict_fing11, augmentdata1), axis=2)
        new_x = new_x + [augmentdata]
        new_y = new_y + [y_train]

    x_train = np.array(new_x)
    y_train = np.array(new_y)





