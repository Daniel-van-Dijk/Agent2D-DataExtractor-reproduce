# new way:
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models, activations, losses, metrics, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

print("VERSION", tensorflow.version.VERSION)
import os
import glob
import matplotlib.pyplot as plt
from numpy import array, random
import numpy as np
import sys
import pandas as pd
run_number = int(sys.argv[1])
run_names = ['', 'X', 'X_F', 'UNUM', 'UNUM_X', 'AKG', 'AKG_F', 'AK', 'AK_F', 'AFC', 'AFC_F']

run_name = f'A{run_number}_alldata_unum'
file_name = f'all_team.csv_{run_names[run_number]}'
k_best = 1
use_pass = False
print(run_name)
use_cluster = True

def get_col_x(header_name_to_num):
    cols = []
    cols.append(['cycle', -1])
    cols.append(['ball_pos_x', -1])
    cols.append(['ball_pos_y', -1])
    cols.append(['ball_pos_r', -1])
    cols.append(['ball_pos_t', -1])
    cols.append(['ball_kicker_x', -1])
    cols.append(['ball_kicker_y', -1])
    cols.append(['ball_kicker_r', -1])
    cols.append(['ball_kicker_t', -1])
    cols.append(['ball_vel_x', -1])
    cols.append(['ball_vel_y', -1])
    cols.append(['ball_vel_r', -1])
    cols.append(['ball_vel_t', -1])
    
    cols.append(['offside_count', -1])
    for i in range(12):
        cols.append(['dribble_angle_0', -1])
        cols.append(['dribble_angle_1', -1])
        cols.append(['dribble_angle_2', -1])
        cols.append(['dribble_angle_3', -1])
        cols.append(['dribble_angle_4', -1])
        cols.append(['dribble_angle_5', -1])
        cols.append(['dribble_angle_6', -1])
        cols.append(['dribble_angle_7', -1])
        cols.append(['dribble_angle_8', -1])
        cols.append(['dribble_angle_9', -1])
        cols.append(['dribble_angle_10', -1])
        cols.append(['dribble_angle_11', -1])
    for s in ['l', 'r']:
        for p in range(1, 9):
            pass
            cols.append(['p_' + s + '_' + str(p) + '_' + 'side', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'unum', -1])
            #cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_id', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_dash_rate', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_effort_max', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_effort_min', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_kickable', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_margin', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_kick_power', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_decay', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_size', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'player_type_speed_max', -1])
            
            cols.append(['p_' + s + '_' + str(p) + '_' + 'body', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'face', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'tackling', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'kicking', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'card', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'pos_x', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'pos_y', -1])
            #cols.append(['p_' + s + '_' + str(p) + '_' + 'rx', -1])
            #cols.append(['p_' + s + '_' + str(p) + '_' + 'ry', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'pos_r', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'pos_t', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'vel_x', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'vel_y', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'vel_r', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'vel_t', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'pos_count', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'vel_count', -1])
            cols.append(['p_' + s + '_' + str(p) + '_' + 'is_kicker', -1])
            if s == 'l':
                pass
		
                cols.append(['p_' + s + '_' + str(p) + '_' + 'in_offside', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'passangle', -1])
                cols.append(['p_' + s + '_' + str(p) + '_' + 'pass_dist', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'nearoppdist', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'angleGCr_angleGCr', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'angleGCt_angleGCt', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'openGAngle_openGAngle', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'stamina', -1])
                #cols.append(['p_' + s + '_' + str(p) + '_' + 'stamina_count', -1])
		
    for c in range(len(cols)):
        cols[c][1] = header_name_to_num[cols[c][0]]
        cols[c] = cols[c][1]
    return cols


def get_col_y(header_name_to_num):
    cols = []
    #cols.append('out_category')
    #cols.append('out_targetx')
    #cols.append('out_or_target_x_dis')
    #cols.append('out_targety')
    #cols.append('out_or_target_y_dis')
    #cols.append('out_or_target_xy_dis')
    cols.append('out_unum')
    #cols.append('out_unum_index')
    #cols.append("out_ball_speed")
    #cols.append("out_ball_dir")
    #cols.append("out_desc")


    cols_numb = []
    for c in range(len(cols)):
        cols_numb.append(header_name_to_num[cols[c]])
    return cols_numb


def read_file(new_file_path):
    #file_path = '/users/grad/etemad/workspace/robo/dataset_all_team/' + file_name
    #file_path = '/home/nader/Documents/merged_data.csv'
    #file_path = '/storage2/dvandijk/Agent2D-DataExtractor/data_sorted/alldata/combined_csv2.csv'
    #file_name = '522matchespass.csv'
    #file_path = '/storage2/dvandijk/modified_data/fullstate/unum/' + file_name

    file = open(new_file_path, 'r')
    lines = file.readlines()[:]
    header = lines[0].split(',')[:-1]
    header_name_to_num = {}
    out_cat_number = 0
    counter = 0
    for h in header:
        header_name_to_num[h] = counter
        if h == 'out_category':
            out_cat_number = counter
        counter += 1
    rows = []
    line_number = 0
    for line in lines[1:]:
        line_number += 1
        row = line.split(',')[:-1]
        #if len(row) != 970:
        #    print('error in line', line_number, len(row))
        f_row = []
        for r in row:
            try:
                f_row.append(float(r))
            except ValueError:
                print(r)

        if use_pass:
            if f_row[out_cat_number] >= 2.0:
                rows.append(f_row)
        else:
            rows.append(f_row)
    return header_name_to_num, rows





filter = 'pass'
file_path = '/storage2/dvandijk/modified_data/fullstate/chunk/' + filter + '/'
os.chdir(file_path)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

chunkbool = True
if chunkbool:
    file_num = 0
    for file in all_filenames:
        file_path2 = file_path + file
        print(file_path2)
        header_name_to_num, rows = read_file(file_path2)
        all_data = array(rows)
        cols = get_col_x(header_name_to_num)
        array_cols = array(cols)
        del cols
        print(array_cols.shape)
        print(all_data.shape)
        data_x = all_data[:, array_cols[:]]

        #print(header_name_to_num)
        cols_numb_y = get_col_y(header_name_to_num)
        array_cols_numb_y = array(cols_numb_y)
        del cols_numb_y
        data_y = (all_data[:, array_cols_numb_y[:]])
        #data_y[:, 0] +
        #if not use_cluster:
        #    data_y[:, 0] /= 3.0
        #    data_y[:, 1] += 180.0
        #    data_y[:, 1] /= 360.0
        del all_data
        data_size = data_x.shape[0]
        train_size = int(data_size * 0.8)

        randomize = np.arange(data_size)
        np.random.shuffle(randomize)
        X = data_x[randomize]
        del data_x
        Y = data_y[randomize]
        del data_y
        train_datas = X[:train_size]
        train_labels = Y[:train_size]
        test_datas = X[train_size + 1:]
        test_labels = Y[train_size + 1:]
        del X
        del Y

        if use_cluster:
            train_labels = to_categorical(train_labels)
            test_labels = to_categorical(test_labels)
            print(train_datas.shape, train_labels.shape)
            print(test_datas.shape, test_labels.shape)

            if file_num == 0:
                print('init')
                network = models.Sequential()
                network.add(layers.Dense(512, activation=activations.relu, input_shape=(train_datas.shape[1],)))
                network.add(layers.Dense(256, activation=activations.relu))
                network.add(layers.Dense(64, activation=activations.relu))
                network.add(layers.Dense(train_labels.shape[1], activation=activations.softmax))

                def accuracy(y_true, y_pred):
                    y_true = K.cast(y_true, y_pred.dtype)
                    y_true = K.argmax(y_true)
                    # y_pred1 = K.argmax(y_pred)
                    res = K.in_top_k(y_pred, y_true, k_best)
                    return res

                network.compile(optimizer=optimizers.Adam(learning_rate=0.0002), loss=losses.categorical_crossentropy, metrics=[accuracy])
            else:

                #from keras.models import load_model
                model_name2 = 'weights' #'best_of_' + str(chunk_num-1)
                print('run2')
                network = models.load_model(model_name2)

            history = network.fit(train_datas, train_labels, epochs=5, batch_size=64, validation_data=(test_datas, test_labels))
            res = network.predict(test_datas)
            #for i in range(len(test_datas)):
            #    print(test_labels[i], res[i])
            history_dict = history.history
            #print(history_dict)
            loss_values = history_dict['loss']
            val_loss_values = history_dict['val_loss']
            acc_values = history_dict['accuracy']
            val_acc_values = history_dict['val_accuracy']
            epochs = range(len(loss_values))
            plt.figure(1)
            plt.subplot(211)
            plt.plot(epochs, loss_values, 'r--', label='Training loss')
            plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')
            plt.title("train/test loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(212)
            plt.plot(epochs, acc_values, 'r--', label='Training accuracy')
            plt.plot(epochs, val_acc_values, '--', label='Validation accuracy')
            plt.title("train/test acc")
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()
            plt.savefig(run_name)
            model_name = 'weights'
            print(model_name)
            network.save(model_name)
            file = open('history_' + run_name, 'w')
            file.write(str(history_dict))
            file.close()
            file = open('best_of_' + str(file) , 'w')
            file.write(str(max(val_acc_values)))
            file.close()
            file_num += 1



        else:
            print(train_datas.shape, train_labels.shape)
            print(test_datas.shape, test_labels.shape)
            network = models.Sequential()
            network.add(layers.Dense(512, activation=activations.relu, input_shape=(train_datas.shape[1],)))
            network.add(layers.Dense(256, activation=activations.relu))
            network.add(layers.Dense(64, activation=activations.relu))
            network.add(layers.Dense(train_labels.shape[1], activation=activations.sigmoid))

            network.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.mse, metrics=[metrics.mse])
            history = network.fit(train_datas, train_labels, epochs=100, batch_size=32,
                                  validation_data=(test_datas, test_labels))

            history_dict = history.history

            loss_values = history_dict['loss']
            val_loss_values = history_dict['val_loss']
            acc_values = history_dict['mean_squared_error']
            val_acc_values = history_dict['val_mean_squared_error']

            epochs = range(len(loss_values))
            plt.figure(1)
            plt.subplot(211)
            plt.plot(epochs, loss_values, 'r--', label='Training loss')
            plt.plot(epochs, val_loss_values, 'b--', label='Validation loss')
            plt.title("train/test loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(212)
            plt.plot(epochs, acc_values, 'r--', label='Training mean_squared_error')
            plt.plot(epochs, val_acc_values, '--', label='Validation mean_squared_error')
            plt.title("train/test acc")
            plt.xlabel('Epochs')
            plt.ylabel('Acc')
            plt.legend()
            plt.savefig(run_name)
            network.save(run_name + '.h5')
            file = open('history_' + run_name, 'w')
            file.write(str(history_dict))
            file.close()
            file = open('best_' + run_name, 'w')
            file.write(str(max(val_acc_values)))
            file.close()
