import struct
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pylab
import torch_geometric

from collections import OrderedDict

print(torch.cuda.is_available())
print(torch.__version__)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GraphConv, GraphSAGE


def EAS_net_Save():
    state = {'net': EAS_net.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'EAS_net.pt')


def EAS_net_Load():
    state = torch.load('EAS_net.pt')
    EAS_net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])


def nomalize(data):
    array = np.array(data)
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return normalized_array.tolist()


def transfer_to_new_times(time):
    for n in range(len(time)):
        try:
            a = list(set(time[n]))
            if min(a) == -1.:
                a.remove(min(a))
            b = min(a)
            time[n] = list(time[n])
            for i in range(len(time[n])):
                if time[n][i] != -1.:
                    time[n][i] = time[n][i] - b
                else:
                    time[n][i] = time[n][i]
        except:
            ...
    return time


def angle_between_vectors(tensor1, tensor2):
    final_tensor = []

    for i in range(len(tensor1)):
        try:
            x1 = tensor1[:, 0].unsqueeze_(1).float()[i].item()
            y1 = tensor1[:, 1].unsqueeze_(1).float()[i].item()
            z1 = tensor1[:, 2].unsqueeze_(1).float()[i].item()

            x2 = tensor2[:, 0].unsqueeze_(1).float()[i].item()
            y2 = tensor2[:, 1].unsqueeze_(1).float()[i].item()
            z2 = tensor2[:, 2].unsqueeze_(1).float()[i].item()

            dot_product = x1 * x2 + y1 * y2 + z1 * z2
            magnitude_1 = math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
            magnitude_2 = math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

            cos_angle = dot_product / (magnitude_1 * magnitude_2)
            angle = math.degrees(math.acos(cos_angle))

            final_tensor.append(angle)
        except ZeroDivisionError:
            final_tensor.append(None)

    final_tensor = torch.tensor(final_tensor, dtype=torch.float32).unsqueeze_(1).float()

    return final_tensor


def x_y_to_r(tensor1, tensor2):
    Rad = []
    for i in range(len(tensor1)):
        x1 = tensor1[:, 2].unsqueeze_(1).float()[i].item()
        y1 = tensor1[:, 3].unsqueeze_(1).float()[i].item()
        x2 = tensor2[:, 2].unsqueeze_(1).float()[i].item()
        y2 = tensor2[:, 3].unsqueeze_(1).float()[i].item()
        r = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        Rad.append(r)
    Radius = torch.tensor(Rad).unsqueeze_(1).float()

    return Radius


def read_binarry(binary_file):
    power = []
    age = []
    coordinate_x = []
    coordinate_y = []
    angle_tetta = []
    angle_phi = []
    energy = []
    time = []

    for i in range(100000):
        q1, p1 = 5, 1
        data_byte = binary_file.read(4 * q1)
        binary_file.tell()
        data_byte = binary_file.read(4 * p1)
        tetta = struct.unpack('f' * p1, data_byte)[0]

        angle_tetta.append(tetta)

        q2, p2 = 0, 1
        data_byte = binary_file.read(4 * q2)
        binary_file.tell()
        data_byte = binary_file.read(4 * p2)
        phi = struct.unpack('f' * p2, data_byte)[0]

        angle_phi.append(phi)

        q3, p3 = 0, 1
        data_byte = binary_file.read(4 * q3)
        binary_file.tell()
        data_byte = binary_file.read(4 * p3)
        x0 = struct.unpack('f' * p3, data_byte)[0]

        coordinate_x.append(x0)

        q4, p4 = 0, 1
        data_byte = binary_file.read(4 * q4)
        binary_file.tell()
        data_byte = binary_file.read(4 * p4)
        y0 = struct.unpack('f' * p4, data_byte)[0]

        coordinate_y.append(y0)

        q5, p5 = 5, 1
        data_byte = binary_file.read(4 * q5)
        binary_file.tell()
        data_byte = binary_file.read(4 * p5)
        power_eas = struct.unpack('f' * p5, data_byte)[0]

        power.append(math.log10(power_eas))

        q6, p6 = 0, 1
        data_byte = binary_file.read(4 * q6)
        binary_file.tell()
        data_byte = binary_file.read(4 * p6)
        age_eas = struct.unpack('f' * p6, data_byte)[0]

        age.append(age_eas)

        i, j = 1565, 36
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

    return power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time  # Возвращает tuple


power = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    power.extend(read_binarry(binary_file_1)[0])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    power.extend(read_binarry(binary_file_1)[0])

age = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    age.extend(read_binarry(binary_file_1)[1])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    age.extend(read_binarry(binary_file_1)[1])

coordinate_x = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    coordinate_x.extend(read_binarry(binary_file_1)[2])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    coordinate_x.extend(read_binarry(binary_file_1)[2])

coordinate_y = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    coordinate_y.extend(read_binarry(binary_file_1)[3])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    coordinate_y.extend(read_binarry(binary_file_1)[3])

angle_tetta = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    angle_tetta.extend(read_binarry(binary_file_1)[4])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    angle_tetta.extend(read_binarry(binary_file_1)[4])

angle_phi = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    angle_phi.extend(read_binarry(binary_file_1)[5])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    angle_phi.extend(read_binarry(binary_file_1)[5])

energy = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    energy.extend(read_binarry(binary_file_1)[6])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    energy.extend(read_binarry(binary_file_1)[6])

time = []
with open('C:/Users/User/PycharmProjects/Binary Files/spe27p_100k_2022_correct.dat', 'rb') as binary_file_1:
    time.extend(read_binarry(binary_file_1)[7])
with open('C:/Users/User/PycharmProjects/Binary Files/spe27f_100k_2022_correct.dat', 'rb') as binary_file_1:
    time.extend(read_binarry(binary_file_1)[7])

energy = nomalize(energy)


def del_zero_arrays(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time):
    zero_indexes = []
    for i in range(len(energy)):
        if energy[i].count(0) == len(energy[i]):
            zero_indexes.append(i)

    zero_indexes.sort(reverse=True)

    for index in zero_indexes:
        del power[index]
        del age[index]
        del coordinate_x[index]
        del coordinate_y[index]
        del angle_tetta[index]
        del angle_phi[index]
        del energy[index]
        del time[index]

    return power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time


power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time = \
    del_zero_arrays(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time)


def del_mines_one_arrays(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time):
    mines_one_indexes = []

    for i in range(len(time)):
        if all(element == -1. for element in time[i]):
            mines_one_indexes.append(i)

    mines_one_indexes.sort(reverse=True)

    for index in mines_one_indexes:
        del power[index]
        del age[index]
        del coordinate_x[index]
        del coordinate_y[index]
        del angle_tetta[index]
        del angle_phi[index]
        del energy[index]
        del time[index]

    return power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time


power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time = \
    del_mines_one_arrays(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time)

energy = [list(t) for t in energy]


def what_the_fuck(array, array_2, array_3, array_4, array_5, array_6, array_7, array_8):
    new_array = []
    for n in range(len(array_7)):
        subarrays = []
        for i in range(0, len(array_7[n]), 4):
            subarray = array_7[n][i:i + 4]
            subarrays.append(subarray)
        new_array.append(subarrays)

    indexes_wtf = []
    for n in range(len(new_array)):
        indexes = []
        for i in range(len(new_array[n])):
            count = sum(1 for element in new_array[n][i] if element != -1.0)
            if count >= 2:
                indexes.append(i)
        indexes_wtf.append(indexes)

    list_true_false = []
    for i in range(len(indexes_wtf)):
        list_true_false.append(len(indexes_wtf[i]))

    for n in range(len(list_true_false)):
        if list_true_false[n] >= 7:
            list_true_false[n] = 1
        else:
            list_true_false[n] = 0

    n_list = [i for i in range(0, len(list_true_false))]

    list_to_del = [n_list[i] for i in range(len(list_true_false)) if list_true_false[i] != 1]

    power_array = [value for index, value in enumerate(array) if index not in list_to_del]

    age_array = [value for index, value in enumerate(array_2) if index not in list_to_del]

    coordinate_x_array = [value for index, value in enumerate(array_3) if index not in list_to_del]

    coordinate_y_array = [value for index, value in enumerate(array_4) if index not in list_to_del]

    angle_tetta_array = [value for index, value in enumerate(array_5) if index not in list_to_del]

    angle_phi_array = [value for index, value in enumerate(array_6) if index not in list_to_del]

    energy_array = [value for index, value in enumerate(array_7) if index not in list_to_del]

    time_array = [value for index, value in enumerate(array_8) if index not in list_to_del]

    return power_array, age_array, coordinate_x_array, coordinate_y_array, \
           angle_tetta_array, angle_phi_array, energy_array, time_array


from tqdm import tqdm

for i in tqdm(range(100)):
    power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time = \
        what_the_fuck(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time)


def make_big_matrix(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi):
    coordinates_array = []

    for i in range(len(angle_tetta)):
        array = []
        array.append(np.sin(np.radians(angle_tetta[i])) * np.cos(np.radians(angle_phi[i])))
        array.append(np.sin(np.radians(angle_tetta[i])) * np.sin(np.radians(angle_phi[i])))
        array.append(np.cos(np.radians(angle_tetta[i])))

        coordinates_array.append(array)

    big_daddy = []

    for i in range(len(power)):
        little_daddy = []

        little_daddy.append(power[i] / 10)
        little_daddy.append(age[i] / 10)
        little_daddy.append(coordinate_x[i] / 100)
        little_daddy.append(coordinate_y[i] / 100)
        little_daddy.append(coordinates_array[i][0])
        little_daddy.append(coordinates_array[i][1])
        little_daddy.append(coordinates_array[i][2])

        big_daddy.append(little_daddy)

    return big_daddy


time = transfer_to_new_times(time)
matrix_all = make_big_matrix(power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi)
combined_list = list(zip(energy, time, matrix_all))
random.shuffle(combined_list)
energy_rand, time_rand, matrix_all_rand = zip(*combined_list)

size_1_energy = int(len(energy) * 0.8)
size_2_energy = int(len(energy) * 0.1)

size_1_time = int(len(time) * 0.8)
size_2_time = int(len(time) * 0.1)

size_1_matrix_all = int(len(matrix_all) * 0.8)
size_2_matrix_all = int(len(matrix_all) * 0.1)

energy = energy_rand[:size_1_energy]
energy_V = energy_rand[size_1_energy:size_1_energy + size_2_energy]
energy_T = energy_rand[size_1_energy + size_2_energy:]

time = time_rand[:size_1_time]
time_V = time_rand[size_1_time:size_1_time + size_2_time]
time_T = time_rand[size_1_time + size_2_time:]

matrix_all = matrix_all_rand[:size_1_matrix_all]
matrix_all_V = matrix_all_rand[size_1_matrix_all:size_1_matrix_all + size_2_matrix_all]
matrix_all_T = matrix_all_rand[size_1_matrix_all + size_2_matrix_all:]

device = torch.device('cuda:0')

edges = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, \
          4, 3, 8, 7, 19, 20, 15, 16, 35, 36, 31, 32, 11, 12, 11, 11, 11, 12, 12, 12, 6, 5, 25, 26, 28, 25, 9, 9, 9, 10,
          10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, \
          17, 18, 19, 20, 16, 15, 31, 32, 36, 35, 3, 4, 7, 8, 6, 5, 25, 28, 14, 9, 12, 11, 1, 4, 2, 3, 4, 1, 6, 7, 8, 5,
          10, 11, 12, 9, 14, 15, 16, 13, 18, 19, 20, 17, 22, 23, 24, 21, 26, \
          27, 28, 25, 30, 31, 32, 29, 34, 35, 36, 33, 33, 34, 1, 2, 14, 13, 30, 29, 23, 22, 21, 24, 22, 21, 36, 33, \
          4, 31, 30, 15, 27, 28, 18, 17, 10, 9, 14, 19, 18, 5, 8, 1, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, \
          25, 26, 13, 14, 30, 29, 24, 21, 23, 22, 33, 34, 1, 2, 28, 27, 10, 9, 12, 15, 22, 21, 11, 10], \
         [2, 3, 4, 1, 6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13, 18, 19, 20, 17, 22, 23, 24, 21, 26, \
          27, 28, 25, 30, 31, 32, 29, 34, 35, 36, 33, 33, 34, 1, 2, 14, 13, 30, 29, 23, 22, 21, 24, 22, 21, 36, 33, \
          4, 31, 30, 15, 27, 28, 18, 17, 10, 9, 14, 19, 18, 5, 8, 1, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, \
          25, 26, 13, 14, 30, 29, 24, 21, 23, 22, 33, 34, 1, 2, 28, 27, 10, 9, 12, 15, 22, 21, 11, 10, 1, 2, 3, 4, 5, 6,
          7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, \
          4, 3, 8, 7, 19, 20, 15, 16, 35, 36, 31, 32, 11, 12, 11, 11, 11, 12, 12, 12, 6, 5, 25, 26, 28, 25, 9, 9, 9, 10,
          10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, \
          17, 18, 19, 20, 16, 15, 31, 32, 36, 35, 3, 4, 7, 8, 6, 5, 25, 28, 14, 9, 12, 11, 1, 4]]

edge_index = torch.tensor(edges) - 1
edge_index.shape

xyz = torch.tensor([[-25.36, 5.89, -6.68],
                    [-37.61, 5.89, -6.68],
                    [-37.61, -7.32, -6.68],
                    [-25.36, -7.32, -6.68],
                    [-25.36, 37.33, -6.68],
                    [-37.61, 37.33, -6.68],
                    [-37.61, 24.14, -6.68],
                    [-25.36, 24.14, -6.68],
                    [6.48, 12.65, 0.00],
                    [-6.87, 12.65, 0.00],
                    [-6.87, -12.63, 0.00],
                    [6.48, -12.63, 0.00],
                    [37.37, 10.96, -14.95],
                    [22.47, 10.96, -15.27],
                    [22.47, -3.95, -15.35],
                    [37.37, -3.95, -15.08],
                    [37.37, 45.57, -16.17],
                    [22.98, 45.57, -16.46],
                    [22.98, 28.48, -16.11],
                    [37.37, 28.48, -16.51],
                    [6.91, -56.16, -16.15],
                    [-2.79, -56.16, -15.91],
                    [-2.79, -70.54, -15.93],
                    [6.91, -70.54, -16.10],
                    [-9.54, 53.62, -17.03],
                    [-20.56, 64.71, -17.80],
                    [-20.56, 53.62, -17.50],
                    [-9.54, 42.54, -16.93],
                    [42.14, -16.80, -15.14],
                    [26.94, -16.80, -15.46],
                    [26.94, -32.10, -15.46],
                    [42.14, -32.10, -14.69],
                    [-25.36, -25.75, -7.33],
                    [-37.26, -25.75, -7.36],
                    [-37.26, -39.54, -7.34],
                    [-25.36, -39.54, -7.28]])


def make_features_for_nodes(energy, time):
    big_list = []
    for i in range(len(energy)):
        little_list = []
        little_list.append(energy[i])
        little_list.append(time[i])
        big_list.append(little_list)
    return big_list


def make_graph(energy, time, matrix_all):
    graph_list = []

    for i in range(len(energy)):
        nodes_graph = make_features_for_nodes(energy[i], time[i])
        nodes_graph = torch.tensor(nodes_graph)

        graph = Data(edge_index=edge_index, x=nodes_graph, pos=xyz,
                     y=torch.tensor([matrix_all[i]], dtype=torch.float32)).to(device)

        graph_list.append(graph)
    return graph_list


graph = make_graph(energy, time, matrix_all)
graph_V = make_graph(energy_V, time_V, matrix_all_V)
graph_T = make_graph(energy_T, time_T, matrix_all_T)

from torch_geometric.loader import DataLoader

num_graphs_per_batch = len(graph)

graph_loader_for_training = DataLoader(graph, batch_size=num_graphs_per_batch, shuffle=False)

graph_loader = DataLoader(graph, batch_size=len(graph), shuffle=False)
graph_loader_V = DataLoader(graph_V, batch_size=len(graph_V), shuffle=False)
graph_loader_T = DataLoader(graph_T, batch_size=len(graph_T), shuffle=False)

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCN(torch.nn.Module):
    def __init__(self, l2_lambda):
        super(GCN, self).__init__()
        torch.manual_seed(0)

        self.initial_conv = GCNConv(2, 4)
        self.conv1 = GCNConv(4, 12)
        self.conv2 = GCNConv(12, 16)
        self.conv3 = GCNConv(16, 32)

        self.fc1 = Linear(32 * 2, 7, bias=True)

        self.drop = torch.nn.Dropout(0.2)

        self.l2_lambda = l2_lambda

    def forward(self, x, edge_index, pos, batch_index):
        hidden = F.relu(self.initial_conv(x, edge_index))

        hidden = F.relu(self.conv1(hidden, edge_index))
        hidden = self.drop(hidden)
        hidden = F.relu(self.conv2(hidden, edge_index))
        hidden = F.relu(self.conv3(hidden, edge_index))

        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)

        out = F.relu(self.fc1(hidden))

        return out, hidden

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_lambda * l2_loss


l2_lambda = 0.001

model = GCN(l2_lambda)
model = model.to(device)
print(model)

import warnings

warnings.filterwarnings("ignore")

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-5, weight_decay=l2_lambda)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1.0e-6, alpha=0.9, eps=1e-08, weight_decay=l2_lambda, momentum=0,
                                centered=False)

losses = []
for epoch in range(20000):

    for batch_for_training in graph_loader_for_training:
        batch_for_training.to(device)
        optimizer.zero_grad()
        pred1, pred2, pred3, pred4, embedding = model(batch_for_training.x.float(), batch_for_training.edge_index,
                                                      batch_for_training.pos, batch_for_training.batch)

        target1 = batch_for_training.y[:, 0].unsqueeze_(1).float()
        target2 = batch_for_training.y[:, 1].unsqueeze_(1).float()
        target3 = batch_for_training.y[:, 2:4]
        target4 = batch_for_training.y[:, 4:7]

        loss1 = criterion(pred1, target1)
        loss2 = criterion(pred2, target2)
        loss3 = criterion(pred3, target3)
        loss4 = criterion(pred4, target4)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()
        optimizer.step()

    losses.append(loss)

    if epoch % 10 == 0:

        print('\n', f"Epoch {epoch}")
        print('\n', "TRAIN--TRAIN--TRAIN--TRAIN--TRAIN", '\n')

        for batch in graph_loader:
            # batch.to(device)
            pred1, pred2, pred3, pred4, embedding = model(batch.x.float(), batch.edge_index,
                                                          batch.pos, batch.batch)

            target1 = batch.y[:, 0].unsqueeze_(1).float()
            target2 = batch.y[:, 1].unsqueeze_(1).float()
            target3 = batch.y[:, 2:4]
            target4 = batch.y[:, 4:7]

            pred1 = pred1
            pred2 = pred2
            pred3 = pred3
            pred4 = pred4

            train_accuracy_power = (10 * pred1 - 10 * target1).std()
            print('POWER', train_accuracy_power)

            train_accuracy_age = (10 * pred2 - 10 * target2).std()
            print('AGE', train_accuracy_age)

            train_accuracy_x = (
                        100 * pred3[:, 0].unsqueeze_(1).float() - 100 * target3[:, 0].unsqueeze_(1).float()).std()
            train_accuracy_y = (
                        100 * pred3[:, 1].unsqueeze_(1).float() - 100 * target3[:, 1].unsqueeze_(1).float()).std()
            print('X', train_accuracy_x)
            print('Y', train_accuracy_y)

            train_accuracy_angle = angle_between_vectors(pred4, target4).std().to(device)
            print('ANGLE', train_accuracy_angle)

        print('\n', "VAL--VAL--VAL--VAL--VAL", '\n')

#        for batch_V in graph_loader_V:
#            batch_V.to(device)
#            pred_V, embedding_V = model(batch_V.x.float(), batch_V.edge_index, batch_V.pos, batch_V.batch)
#
#            target_V = batch_V.y.to(device)
#            pred_V = pred_V.to(device)
#
#            val_accuracy_power = (10 * (target_V[:, 0].unsqueeze_(1).float()) - 10 * (pred_V[:, 0].unsqueeze_(1).float())).std()
#            print('POWER', val_accuracy_power)
#
#            val_accuracy_age = (target_V[:, 1].unsqueeze_(1).float() - pred_V[:, 1].unsqueeze_(1).float()).std()
#            print('AGE', val_accuracy_age)
#
#            val_accuracy_x = (100 * (target_V[:, 2].unsqueeze_(1).float()) - 100 * (pred_V[:, 2].unsqueeze_(1).float())).std()
#            val_accuracy_y = (100 * (target_V[:, 3].unsqueeze_(1).float()) - 100 * (pred_V[:, 3].unsqueeze_(1).float())).std()
#            print('X', val_accuracy_x)
#            print('Y', val_accuracy_y)
#
#            val_accuracy_angle = angle_between_vectors(target_V, pred_V).std().to(device)
#            print('ANGLE', val_accuracy_angle)