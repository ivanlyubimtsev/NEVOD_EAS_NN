import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pylab

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

def EAS_net_Save():
    state = {'net': EAS_net.state_dict(),
    'optimizer': optimizer.state_dict()}
    torch.save(state, 'EAS_net.pt')

def EAS_net_Load():
    state = torch.load('EAS_net.pt')
    EAS_net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])

def convert_to_x(phi, tetta):
    x = math.sin(math.radians(tetta)) * math.cos(math.radians(phi))
    return x

def convert_to_y(phi, tetta):
    y = math.sin(math.radians(tetta)) * math.sin(math.radians(phi))
    return y

def convert_to_z(tetta):
    z = math.cos(math.radians(tetta))
    return z

def transfer_to_new_times(threshold_time):
    try:
        a = list(set(threshold_time))
        a.remove(min(a))
        a = min(a)
        threshold_time = list(threshold_time)
        for i in range(len(threshold_time)):
            if threshold_time[i] > 0:
                threshold_time[i] = threshold_time[i] - a
            else:
                threshold_time[i] = threshold_time[i]
    except:
        ...
    return threshold_time

def angle_bwn_2vectors(coordinates, preds):
    x1 = coordinates[0]
    y1 = coordinates[1]
    z1 = coordinates[2]
    x2 = preds[0]
    y2 = preds[1]
    z2 = preds[2]
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    length_1 = torch.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
    length_2 = torch.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
    cos_angle = dot_product / (length_1 * length_2)
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg

with open('C:/Users/lia/PycharmProjects/Neural Networks/spe27p_100k_2022_correct.dat', 'rb') as binary_file:

    energy = []
    time = []
    energy_plus_time = []
    coordinates = []

    energy_V = []
    time_V = []
    energy_plus_time_V = []
    coordinates_V = []

    energy_T = []
    time_T = []
    energy_plus_time_T = []
    coordinates_T = []

    # Считываю бинарник для тренировочной выборки

    for i in range(80000):
        q1, p1 = 5, 1
        data_byte = binary_file.read(4 * q1)
        binary_file.tell()
        data_byte = binary_file.read(4 * p1)
        tetta = struct.unpack('f' * p1, data_byte)[0]  # * 0.0174444

        q2, p2 = 0, 1
        data_byte = binary_file.read(4 * q2)
        binary_file.tell()
        data_byte = binary_file.read(4 * p2)
        phi = struct.unpack('f' * p2, data_byte)[0]  # * 0.0174444

        wtf = []
        wtf.append(convert_to_x(phi, tetta))
        wtf.append(convert_to_y(phi, tetta))
        wtf.append(convert_to_z(tetta))
        coordinates.append(wtf)

        q3, p3 = 0, 1
        data_byte = binary_file.read(4 * q3)
        binary_file.tell()
        data_byte = binary_file.read(4 * p3)
        x0 = struct.unpack('f' * p3, data_byte)[0]

        q4, p4 = 0, 1
        data_byte = binary_file.read(4 * q4)
        binary_file.tell()
        data_byte = binary_file.read(4 * p4)
        y0 = struct.unpack('f' * p4, data_byte)[0]


        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow = struct.unpack('f' * p0, data_byte)[0]
        #power.append(math.log10(pow))

        i, j = 1566, 36
        data_byte = binary_file.read(4 * i)
        binary_file.tell()
        data_byte = binary_file.read(4 * j)
        energy_release = struct.unpack('f' * j, data_byte)
        energy.append(energy_release)

        k, m = 1, 144
        data_byte = binary_file.read(4 * k)
        binary_file.tell()
        data_byte = binary_file.read(4 * m)
        t = struct.unpack('f' * m, data_byte)
        threshold_time = t[::4]
        time.append(threshold_time)

    #energy = nomalize(energy)

    time = transfer_to_new_times(time)
    #time, x = without_zero_events(time, x)
    #time = remove_mines_to_zero(time)
    energy = torch.tensor(energy)
    time = torch.tensor(time)

    energy_plus_time = torch.cat((energy, time), 1)
    coordinates = torch.tensor(coordinates).float()

    # Считываю бинарник для валидационной выборки

    for m in range(10000):
        q1, p1 = 5, 1
        data_byte = binary_file.read(4 * q1)
        binary_file.tell()
        data_byte = binary_file.read(4 * p1)
        tetta_V = struct.unpack('f' * p1, data_byte)[0]  # * 0.0174444

        q2, p2 = 0, 1
        data_byte = binary_file.read(4 * q2)
        binary_file.tell()
        data_byte = binary_file.read(4 * p2)
        phi_V = struct.unpack('f' * p2, data_byte)[0]  # * 0.0174444

        wtf_V = []
        wtf_V.append(convert_to_x(phi_V, tetta_V))
        wtf_V.append(convert_to_y(phi_V, tetta_V))
        wtf_V.append(convert_to_z(tetta_V))
        coordinates_V.append(wtf_V)

        q3, p3 = 0, 1
        data_byte = binary_file.read(4 * q3)
        binary_file.tell()
        data_byte = binary_file.read(4 * p3)
        x0_V = struct.unpack('f' * p3, data_byte)[0]

        q4, p4 = 0, 1
        data_byte = binary_file.read(4 * q4)
        binary_file.tell()
        data_byte = binary_file.read(4 * p4)
        y0_V = struct.unpack('f' * p4, data_byte)[0]

        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow_V = struct.unpack('f' * p0, data_byte)[0]
        #power_V.append(math.log10(pow_V))

        i, j = 1566, 36
        data_byte = binary_file.read(4 * i)
        binary_file.tell()
        data_byte = binary_file.read(4 * j)
        energy_release_V = struct.unpack('f' * j, data_byte)
        energy_V.append(energy_release_V)

        k, m = 1, 144
        data_byte = binary_file.read(4 * k)
        binary_file.tell()
        data_byte = binary_file.read(4 * m)
        t_V = struct.unpack('f' * m, data_byte)
        threshold_time_V = t_V[::4]
        time_V.append(threshold_time_V)

    #energy_V = nomalize(energy_V)

    time_V = transfer_to_new_times(time_V)
    #time_V, x_V = without_zero_events(time_V, x_V)
    #time_V = remove_mines_to_zero(time_V)
    energy_V = torch.tensor(energy_V)
    time_V = torch.tensor(time_V)

    energy_plus_time_V = torch.cat((energy_V, time_V), 1)
    coordinates_V = torch.tensor(coordinates_V).float()

    # Считываю бинарник для тестовой выборки

    for n in range(10000):
        q1, p1 = 5, 1
        data_byte = binary_file.read(4 * q1)
        binary_file.tell()
        data_byte = binary_file.read(4 * p1)
        tetta_T = struct.unpack('f' * p1, data_byte)[0]  # * 0.0174444

        q2, p2 = 0, 1
        data_byte = binary_file.read(4 * q2)
        binary_file.tell()
        data_byte = binary_file.read(4 * p2)
        phi_T = struct.unpack('f' * p2, data_byte)[0]  # * 0.0174444

        wtf_T = []
        wtf_T.append(convert_to_x(phi_T, tetta_T))
        wtf_T.append(convert_to_y(phi_T, tetta_T))
        wtf_T.append(convert_to_z(tetta_T))
        coordinates_T.append(wtf_T)

        q3, p3 = 0, 1
        data_byte = binary_file.read(4 * q3)
        binary_file.tell()
        data_byte = binary_file.read(4 * p3)
        x0_T = struct.unpack('f' * p3, data_byte)[0]

        q4, p4 = 0, 1
        data_byte = binary_file.read(4 * q4)
        binary_file.tell()
        data_byte = binary_file.read(4 * p4)
        y0_T = struct.unpack('f' * p4, data_byte)[0]


        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow_T = struct.unpack('f' * p0, data_byte)[0]
        #power_T.append(math.log10(pow_T))

        i, j = 1566, 36
        data_byte = binary_file.read(4 * i)
        binary_file.tell()
        data_byte = binary_file.read(4 * j)
        energy_release_T = struct.unpack('f' * j, data_byte)
        energy_T.append(energy_release_T)

        k, m = 1, 144
        data_byte = binary_file.read(4 * k)
        binary_file.tell()
        data_byte = binary_file.read(4 * m)
        t_T = struct.unpack('f' * m, data_byte)
        threshold_time_T = t_T[::4]
        time_T.append(threshold_time_T)

    #energy_T = nomalize(energy_T)

    time_T = transfer_to_new_times(time_T)
    #time_T, x_T = without_zero_events(time_T, x_T)
    #time_T = remove_mines_to_zero(time_T)
    energy_T = torch.tensor(energy_T)
    time_T = torch.tensor(time_T)

    energy_plus_time_T = torch.cat((energy_T, time_T), 1)
    coordinates_V = torch.tensor(coordinates_V).float()

class EAS_net(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, num_hidden_3, num_hidden_4,
                 num_hidden_5, num_hidden_6, num_hidden_7, num_outputs, p):

        super(EAS_net, self).__init__()

        self.EAS_net = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(num_hidden_1, num_hidden_2),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p),
            torch.nn.Linear(num_hidden_2, num_hidden_3),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_3, num_hidden_4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(num_hidden_4, num_hidden_5),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p),
            torch.nn.Linear(num_hidden_5, num_hidden_6),
            torch.nn.ReLU(),
            torch.nn.Dropout(p),
            torch.nn.Linear(num_hidden_6, num_hidden_7),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_7, num_outputs)
        )

    def forward(self, x):
        x = self.EAS_net(x)
        return x

EAS_net = EAS_net(72,
                  64, 32, 32, 24, 16, 8, 4,
                  3,
                  0.5)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EAS_net = EAS_net.to(device)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(EAS_net.parameters(), lr=3*1.0e-4)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1, verbose=False)

val_loss_history = []
train_loss_history = []
val_accuracy_history = []
train_accuracy_history = []

print('Нейронка начинает работать :)')

batch_size = 256

for epoch in range(400):
    order = np.random.permutation(len(energy_plus_time))

    for start_index in range(0, len(energy_plus_time), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        energy_plus_time_batch = energy_plus_time[batch_indexes].to(device)
        coordinates_batch = coordinates[batch_indexes].to(device)

        preds = EAS_net.forward(energy_plus_time_batch)

        loss_value = loss(preds, coordinates_batch)

        loss_value.backward()

        optimizer.step()
    #scheduler.step()

    preds = EAS_net.forward(energy_plus_time)
    val_preds = EAS_net.forward(energy_plus_time_V)

    train_loss_history.append(loss(preds, coordinates).item())
    val_loss_history.append(loss(val_preds, coordinates_V).item())

    train_accuracy = torch.std(angle_bwn_2vectors(coordinates, preds), axis=-1)
    val_accuracy = torch.std(angle_bwn_2vectors(coordinates_V, val_preds), axis=-1)
    #train_accuracy = (angle_bwn_2vectors(coordinates, preds)).std()
    #val_accuracy = (angle_bwn_2vectors(coordinates_V, val_preds)).std()

    train_accuracy_history.append(train_accuracy.item())
    val_accuracy_history.append(val_accuracy.item())

    print(epoch, '\n', train_accuracy, '\n', val_accuracy, '\n', optimizer.param_groups[-1]['lr'])

EAS_net_Save()