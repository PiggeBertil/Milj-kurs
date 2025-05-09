# Task 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "data/utslappRCP45.csv"
emissions_df = pd.read_csv(filename)

filename = "data/koncentrationerRCP45.csv"
concentrations_df = pd.read_csv(filename)

time = concentrations_df["Time (year)"].values

B_0 = np.array([600.0, 600.0, 1500.0]) # before industrial
Bs = [B_0] # after industrial

F_0 = np.zeros([3,3]) # before industrial

F_0[0,1] = 60.0 # Box 1 --> 2
F_0[1,0] = 15.0 # Box 2 --> 1
F_0[1,2] = 45.0 # Box 2 --> 3
F_0[2,0] = 45.0 # Box 3 --> 1

F = F_0 # after industrial

beta = 0.35

conversion_factor = 0.469 # ppm CO2/GtC

alpha = F_0[:,:] / B_0[:, np.newaxis]

NPP_0 = 60 # Flow from box 1 to box 2 before industrial

for t in time[1:]:
    B = Bs[-1]

    NPP = NPP_0 * (1 + beta * np.log(B[0]/B_0[0]))
    U = emissions_df.loc[emissions_df["Time (year)"] == t].values[0][1]

    dB = np.zeros(3)
    dB[0] = alpha[2,0] * B[2] + alpha[1,0] * B[1] - NPP + U
    dB[1] = NPP - alpha[1,2] * B[1] - alpha[1,0] * B[1]
    dB[2] = alpha[1,2] * B[1] - alpha[2,0] * B[2]

    Bs.append(B + dB)

print(Bs)

plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "].values)
plt.plot(time, [B[0] * conversion_factor for B in Bs])
plt.show()

# Task 2
# larger beta ==> larger NPP ==> smaller delta B1 and larger delta B2
# larger B2 ==> larger delta B3