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

with open('spe27p_100k_2022_correct.dat', 'rb') as binary_file:
    angle_phi = []
    angle_tetta = []
    coordinates_x = []
    coordinates_y = []
    power = []
    energy = []

    angle_phi_V = []
    angle_tetta_V = []
    coordinates_x_V = []
    coordinates_y_V = []
    power_V = []
    energy_V = []

    angle_phi_T = []
    angle_tetta_T = []
    coordinates_x_T = []
    coordinates_y_T = []
    power_T = []
    energy_T = []

    #Считываю бинарник для тренировочной выборки

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
        time = struct.unpack('f' * m, data_byte)
        threshold_time = time[::4]

    power = torch.tensor(power)
    power = power.unsqueeze_(1).float()
    energy = torch.tensor(energy)
    coordinates_x = torch.tensor(coordinates_x).unsqueeze_(1).float()
    coordinates_y = torch.tensor(coordinates_y).unsqueeze_(1).float()

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
        time_V = struct.unpack('f' * m, data_byte)
        threshold_time_V = time_V[::4]

    power_V = torch.tensor(power_V)
    power_V = power_V.unsqueeze_(1).float()
    energy_V = torch.tensor(energy_V)
    coordinates_x_V = torch.tensor(coordinates_x_V).unsqueeze_(1).float()
    coordinates_y_V = torch.tensor(coordinates_y_V).unsqueeze_(1).float()

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
        time_T = struct.unpack('f' * m, data_byte)
        threshold_time_T = time_T[::4]

    power_T = torch.tensor(power_T)
    power_T = power_T.unsqueeze_(1).float()
    energy_T = torch.tensor(energy_T)
    coordinates_x_T = torch.tensor(coordinates_x_T).unsqueeze_(1).float()
    coordinates_y_T = torch.tensor(coordinates_y_T).unsqueeze_(1).float()


class EAS_net(torch.nn.Module):
    def __init__(self):
        super(EAS_net, self).__init__()

        self.fc1 = torch.nn.Linear(36, 128)
        self.ac1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 55)
        self.ac2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(55, 66)
        self.ac3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(66, 33)
        self.ac4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(33, 22)
        self.ac5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(22, 11)
        self.ac6 = torch.nn.ReLU()
        self.fc7 = torch.nn.Linear(11, 1)

    def forward(self, x):

        #print(x.shape)
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.fc4(x)
        x = self.ac4(x)
        x = self.fc5(x)
        x = self.ac5(x)
        x = self.fc6(x)
        x = self.ac6(x)
        x = self.fc7(x)
        #print(x.shape)

        return x

#Функции для сохранения/загрузки обученной сети

def statSave():
    state = {'net': EAS_net.state_dict(),
    'optimizer': optimizer.state_dict()}
    torch.save(state, 'EAS_net.pt')

def statLoad():
    state = torch.load('EAS_net.pt')
    EAS_net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['optimizer'])

EAS_net = EAS_net()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EAS_net = EAS_net.to(device)

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(EAS_net.parameters(), lr=3*1.0e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,120,150], gamma=0.9,
                                                 #last_epoch=-1, verbose=False)

val_loss_history = []
train_loss_history = []
val_accuracy_history = []
train_accuracy_history = []

print('Нейронка начинает работать :)')

batch_size = 256

for epoch in range(90):
    order = np.random.permutation(len(energy))

    for start_index in range(0, len(energy), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        energy_batch = energy[batch_indexes].to(device)
        power_batch = power[batch_indexes].to(device)

        preds = EAS_net.forward(energy_batch)

        loss_value = loss(preds, power_batch)

        loss_value.backward()

        optimizer.step()
    #scheduler.step()

    preds = EAS_net.forward(energy)
    val_preds = EAS_net.forward(energy_V)

    train_loss_history.append(loss(preds, power).item())
    val_loss_history.append(loss(val_preds, power_V).item())

    train_accuracy = torch.std(power - preds, axis=0)
    val_accuracy = torch.std(power_V - val_preds, axis=0)
    #train_accuracy = (power - preds).std()
    #val_accuracy = (power_V - val_preds).std()

    train_accuracy_history.append(train_accuracy.item())
    val_accuracy_history.append(val_accuracy.item())

    a = torch.mean(power).item()
    b = torch.mean(preds).item()

    print(epoch, '\n', train_accuracy, '\n', val_accuracy, '\n', ((a - b) / a), '\n', optimizer.param_groups[-1]['lr'])

statSave()

#Графическое представление точности, функций потерь и гистограммы разности реальных и предсказанных событий

statLoad()

preds = EAS_net.forward(energy_T)
preds = preds.T
preds = preds[0].tolist()
power_T = power_T.T
power_T = power_T[0].tolist()

subtract = np.subtract(power_T, preds).tolist()
fig, ax = plt.subplots()
#pylab.subplot (2, 1, 2)
plt.hist(subtract, bins=150)
plt.xlabel('Разность моделированных и восстановленных значений десятичного логарифма мощности ШАЛ', fontsize=20)
plt.ylabel('Количество событий', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

subtract = np.subtract(power_T, preds).tolist()
fig, ax = plt.subplots()
#pylab.subplot (2, 1, 2)
plt.hist(power_T, bins=150)
plt.xlabel('Моделированные значения десятичного логарифма мощности ШАЛ', fontsize=20)
plt.ylabel('Количество событий', fontsize=20)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)


#pylab.rcParams['figure.figsize'] = (30.0, 10.0)

#pylab.subplot (2, 2, 1)
#plt.plot(val_accuracy_history, color='blue', label='Validation')
#plt.plot(train_accuracy_history, color='red', label='Train')
#pylab.title ("Accuracy History")
#plt.xlabel('Количество эпох')
#plt.ylabel('Точность(std)')
#pylab.rcParams['figure.figsize'] = (30.0, 10.0)
#plt.legend()

#pylab.subplot (2, 2, 2)
#plt.plot(val_loss_history, color='blue', label='Validation')
#plt.plot(train_loss_history, color='red', label='Train')
#pylab.title ("Loss Function History")
#plt.xlabel('Количество эпох')
#plt.ylabel('Значение loss функции')
#pylab.rcParams['figure.figsize'] = (30.0, 10.0)
#plt.legend()

pylab.show()