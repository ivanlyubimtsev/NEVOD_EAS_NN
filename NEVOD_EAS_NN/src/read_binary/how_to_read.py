# FUNCTION TO READ A BINARY FILE NEVOD-SHAL
# TAKES A .dat FILE AS INPUT

def read_binarry(binary_file):
# ARRAYS ARE CREATED FOR REQUIRED SHAL PARAMETERS (POWER, AGE, COORDINATES OF SHAL AXIS, ZENITH AND AZIMUTH ANGLES, ENERGY RELEASE AT THE STATION, AND RESPONSE TIME OF THE STATION)
    power = []
    age = []
    coordinate_x = []
    coordinate_y = []
    angle_tetta = []
    angle_phi = []
    energy = []
    time = []

    for i in range(100000): # range(100000) - THERE ARE 100K EVENTS IN THE BINARY FILE
        # READ THE FIRST 5 ELEMENTS, THEN READ AND STORE THE REQUIRED ELEMENT IN THE ARRAY
        q1, p1 = 5, 1
        # SET TO 4 DUE TO THE STRUCTURE OF THE BINARY FILE
        data_byte = binary_file.read(4 * q1)
        # SET A CHECKPOINT
        binary_file.tell()
        data_byte = binary_file.read(4 * p1)
        # UNPACK THE FILE ELEMENT
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

    return power, age, coordinate_x, coordinate_y, angle_tetta, angle_phi, energy, time # Returns a tuple
