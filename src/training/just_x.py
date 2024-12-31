import struct
import torch
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

#Функции для нормализации/денормализации для [0,1],
#для преобразования углов и сохранения/загрузки обученной сети

def EAS_net_Save():
    state = {'net': EAS_net.state_dict(),
    'optimizer': optimizer.state_dict()}
    torch.save(state, 'EAS_net.pt')

def EAS_net_Load():
    state = torch.load('EAS_net.pt')
    EAS_net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])

def nomalize(x):
    x = np.array(x)
    x = ((x-np.amin(x))/(np.amax(x)-np.amin(x))).tolist()
    x = torch.tensor(x)
    return x

def denomalize(x, xmin, xmax):
    x = x.tolist()
    x = np.array(x)
    x = x*(xmax - xmin) + xmin
    x = torch.tensor(x)
    return x

def convert_to_x(phi, tetta):
    x = np.sin(tetta)*np.cos(phi)
    return x

def convert_to_y(phi, tetta):
    y = np.sin(tetta)*np.sin(phi)
    return y

def convert_to_z(tetta):
    z = np.cos(tetta)
    return z

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

def without_zero_events(K, B): #K - time, B - coordinates
    A =[]
    C =[]
    length = len(K)
    for i in range(length):
        schetchik = 0
        for j in range(len(K[i])):
            if K[i][j] == -1:
                schetchik = 1 + schetchik
        if schetchik == len(K[i]):
            continue
        else:
            A.append(K[i])
            C.append(B[i])
    return A, C

def remove_mines_to_zero(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j]==-1.:
                x[i][j] = 0.
    return x

with open('C:/Users/lia/PycharmProjects/НИР/_spe27p_100k_2022_correct.dat', 'rb') as binary_file:
    angle_phi = []
    angle_tetta = []
    coordinates_x = []
    coordinates_y = []
    power = []
    energy = []
    time = []
    energy_plus_time = []
    x = []
    y = []
    z = []

    angle_phi_V = []
    angle_tetta_V = []
    coordinates_x_V = []
    coordinates_y_V = []
    power_V = []
    energy_V = []
    time_V = []
    energy_plus_time_V = []
    x_V = []
    y_V = []
    z_V = []

    angle_phi_T = []
    angle_tetta_T = []
    coordinates_x_T = []
    coordinates_y_T = []
    power_T = []
    energy_T = []
    time_T = []
    energy_plus_time_T = []
    x_T = []
    y_T = []
    z_T = []

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

        angle_tetta.append(tetta)
        angle_phi.append(phi)

        x.append(convert_to_x(phi, tetta))
        y.append(convert_to_y(phi, tetta))
        z.append(convert_to_z(tetta))

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

        coordinates_x.append(x0)
        coordinates_y.append(y0)

        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow = struct.unpack('f' * p0, data_byte)[0]
        power.append(math.log10(pow))

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

    energy = nomalize(energy)

    time = transfer_to_new_times(time)
    time, x = without_zero_events(time, x)
    #time = remove_mines_to_zero(time)
    time = torch.tensor(time)

    #energy_plus_time = torch.cat((energy, time), 1)

    power = torch.tensor(power).unsqueeze_(1).float()

    coordinates_x = torch.tensor(coordinates_x).unsqueeze_(1).float()
    coordinates_y = torch.tensor(coordinates_y).unsqueeze_(1).float()

    angle_tetta = torch.tensor(angle_tetta).unsqueeze_(1).float()
    angle_phi = torch.tensor(angle_phi).unsqueeze_(1).float()

    x = torch.tensor(x).unsqueeze_(1).float()
    y = torch.tensor(y).unsqueeze_(1).float()
    z = torch.tensor(z).unsqueeze_(1).float()

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

        angle_tetta_V.append(tetta_V)
        angle_phi_V.append(phi_V)

        x_V.append(convert_to_x(phi_V, tetta_V))
        y_V.append(convert_to_y(phi_V, tetta_V))
        z_V.append(convert_to_z(tetta_V))

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

        coordinates_x_V.append(x0_V)
        coordinates_y_V.append(y0_V)

        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow_V = struct.unpack('f' * p0, data_byte)[0]
        power_V.append(math.log10(pow_V))

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

    energy_V = nomalize(energy_V)

    time_V = transfer_to_new_times(time_V)
    time_V, x_V = without_zero_events(time_V, x_V)
    #time_V = remove_mines_to_zero(time_V)
    time_V = torch.tensor(time_V)

    #energy_plus_time_V = torch.cat((energy_V, time_V), 1)

    power_V = torch.tensor(power_V).unsqueeze_(1).float()

    coordinates_x_V = torch.tensor(coordinates_x_V).unsqueeze_(1).float()
    coordinates_y_V = torch.tensor(coordinates_y_V).unsqueeze_(1).float()

    angle_tetta_V = torch.tensor(angle_tetta_V).unsqueeze_(1).float()
    angle_phi_V = torch.tensor(angle_phi_V).unsqueeze_(1).float()

    x_V = torch.tensor(x_V).unsqueeze_(1).float()
    y_V = torch.tensor(y_V).unsqueeze_(1).float()
    z_V = torch.tensor(z_V).unsqueeze_(1).float()

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

        angle_tetta_T.append(tetta_T)
        angle_phi_T.append(phi_T)

        x_T.append(convert_to_x(phi_T, tetta_T))
        y_T.append(convert_to_y(phi_T, tetta_T))
        z_T.append(convert_to_z(tetta_T))

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

        coordinates_x_T.append(x0_T)
        coordinates_y_T.append(y0_T)

        q0, p0 = 5, 1
        data_byte = binary_file.read(4 * q0)
        binary_file.tell()
        data_byte = binary_file.read(4 * p0)
        pow_T = struct.unpack('f' * p0, data_byte)[0]
        power_T.append(math.log10(pow_T))

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

    energy_T = nomalize(energy_T)

    time_T = transfer_to_new_times(time_T)
    time_T, x_T = without_zero_events(time_T, x_T)
    #time_T = remove_mines_to_zero(time_T)
    time_T = torch.tensor(time_T)

    #energy_plus_time_T = torch.cat((energy_T, time_T), 1)

    power_T = torch.tensor(power_T).unsqueeze_(1).float()

    coordinates_x_T = torch.tensor(coordinates_x_T).unsqueeze_(1).float()
    coordinates_y_T = torch.tensor(coordinates_y_T).unsqueeze_(1).float()

    angle_tetta_T = torch.tensor(angle_tetta_T).unsqueeze_(1).float()
    angle_phi_T = torch.tensor(angle_phi_T).unsqueeze_(1).float()

    x_T = torch.tensor(x_T).unsqueeze_(1).float()
    y_T = torch.tensor(y_T).unsqueeze_(1).float()
    z_T = torch.tensor(z_T).unsqueeze_(1).float()

print(power)

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

EAS_net = EAS_net(36,
                  64, 32, 32, 24, 16, 8, 4,
                  1,
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

for epoch in range(15):
    order = np.random.permutation(len(time))

    for start_index in range(0, len(time), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        time_batch = time[batch_indexes].to(device)
        x_batch = x[batch_indexes].to(device)

        preds = EAS_net.forward(time_batch)

        loss_value = loss(preds, x_batch)

        loss_value.backward()

        optimizer.step()
    #scheduler.step()

    preds = EAS_net.forward(time)
    val_preds = EAS_net.forward(time_V)

    train_loss_history.append(loss(preds, x).item())
    val_loss_history.append(loss(val_preds, x_V).item())

    train_accuracy = (x - preds).std()
    val_accuracy = (x_V - val_preds).std()

    train_accuracy_history.append(train_accuracy.item())
    val_accuracy_history.append(val_accuracy.item())

    print(epoch, '\n', train_accuracy, '\n', val_accuracy, '\n', optimizer.param_groups[-1]['lr'])

EAS_net_Save()

#Графическое представление точности, функций потерь и гистограммы разности реальных и предсказанных событий

EAS_net_Load()

preds = EAS_net.forward(time_T)
preds = preds.T
preds = preds[0].tolist() #Why 0
x_T = x_T.T
x_T = x_T[0].tolist()
subtract = np.subtract(x_T, preds).tolist()

pylab.subplot (2, 1, 2)
plt.hist(subtract, bins=300)
plt.xlabel('Разность реальных и предсказанных значений')
plt.ylabel('Количество событий')
pylab.rcParams['figure.figsize'] = (30.0, 10.0)

pylab.subplot (2, 2, 1)
plt.plot(val_accuracy_history, color='blue', label='Validation')
plt.plot(train_accuracy_history, color='red', label='Train')
pylab.title ("Accuracy History")
plt.xlabel('Количество эпох')
plt.ylabel('Точность(std)')
pylab.rcParams['figure.figsize'] = (30.0, 10.0)
plt.legend()

pylab.subplot (2, 2, 2)
plt.plot(val_loss_history, color='blue', label='Validation')
plt.plot(train_loss_history, color='red', label='Train')
pylab.title ("Loss Function History")
plt.xlabel('Количество эпох')
plt.ylabel('Значение loss функции')
pylab.rcParams['figure.figsize'] = (30.0, 10.0)
plt.legend()

pylab.show()