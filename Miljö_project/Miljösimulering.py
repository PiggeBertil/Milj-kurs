import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = 'data/koncentrationerRCP45.csv'
concentrations_df = pd.read_csv(filename)

filename = 'data/utslappRCP45.csv'
emisions_df = pd.read_csv(filename)



time = concentrations_df["Time (year)"].values

B_0 = np.array([600.0, 600.0, 1500.0])  # GtC



Conversion_factor = 0.469 # ppm CO2/GtC


def simulate(BETA = 0.3, B_0=B_0, time = time):


    Bs = [B_0] # GtC

    F_0 = np.zeros((3, 3))

    NPP_0 = 60.0

    F_0[1,0] = 15.0
    F_0[1,2] = 45.0
    F_0[2,0] = 45.0
    F_0[0,1] = NPP_0

    F = F_0

    alpha = F_0[:, :] / B_0[:, np.newaxis]

    for t in time[1:]:
        B = Bs[-1]

        NPP = NPP_0 * (1 + BETA*np.log(B[0]/B_0[0]))

        U = emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1]

        dB = np.zeros(3)


        dB[0] = alpha[2,0]*B[2] + alpha[1,0]*B[1] - NPP + U
        dB[1] = NPP - alpha[1,2]*B[1] - alpha[1,0]*B[1]
        dB[2] = alpha[1,2]*B[1] - alpha[2,0]*B[2]
        
        Bs.append(B + dB)

    return Bs
    


    
Bs = simulate(BETA=0.1)
plt.plot(time, [B[0]*Conversion_factor for B in Bs], label= "B_1 Beta=0.1", linestyle = "dotted", color = "red")
plt.plot(time, [B[1]*Conversion_factor for B in Bs], label= "B_2 Beta=0.1", linestyle = "dotted", color = "blue")
plt.plot(time, [B[2]*Conversion_factor for B in Bs], label= "B_3 Beta=0.1", linestyle = "dotted", color = "green")

Bs = simulate(BETA=0.35)
plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "])
plt.plot(time, [B[0]*Conversion_factor for B in Bs], label= "B_1 Beta=0.35", color = "red")
plt.plot(time, [B[1]*Conversion_factor for B in Bs], label= "B_2 Beta=0.35", color = "blue")
plt.plot(time, [B[2]*Conversion_factor for B in Bs], label= "B_3 Beta=0.35", color = "green")

Bs = simulate(BETA=0.8)
plt.plot(time, [B[0]*Conversion_factor for B in Bs], label= "B_1 Beta=0.8", color = "red", linestyle="dashed")
plt.plot(time, [B[1]*Conversion_factor for B in Bs], label= "B_2 Beta=0.8", color = "blue", linestyle="dashed")
plt.plot(time, [B[2]*Conversion_factor for B in Bs], label= "B_3 Beta=0.8", color = "green", linestyle="dashed")


plt.legend()

plt.show()

# Task 3

def I(t, U_cum):

    k = 3.06 * 10 ** (-3)
    summa = 0
    for i in range(5):
        tau_i = tau_0[i] * (1 + k * U_cum)
        summa += A[i] * np.exp(-t/tau_i)
    return summa

A = np.array([0.113, 0.213, 0.258, 0.273, 0.1430])
tau_0 = np.array([2.0, 12.2, 50.4, 243.3, np.inf])




U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])
cumulative_emissions = np.cumsum(U)
cumulative_emissions_shifted = np.concatenate(([0], cumulative_emissions[:-1])) # summan är upp till t-1


def I(t, time):

    k = 3.06 * 10 ** (-3)
    summa = 0
    for i in range(5):
        tau_i = tau_0[i]*(1+k*cumulative_emissions_shifted[time])
        summa += A[i] * np.exp(-t/tau_i)
    return summa



plt.plot([I(t, 100) for t in range(500)])
plt.plot([I(t, 150) for t in range(500)])
plt.plot([I(t, 200) for t in range(500)])
plt.plot([I(t, 250) for t in range(500)])
plt.plot([I(t, 300) for t in range(500)])

plt.show()



"""
k = 3.06 * 10 ** (-3)


U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])
cumulative_emissions = np.cumsum(U)
cumulative_emissions_shifted = np.concatenate(([0], cumulative_emissions[:-1])) # summan är upp till t-1

tau_all = np.zeros((len(time), 5))
for i in range(5):
    tau_all[:,i] = tau_0[i] * (1 + k * cumulative_emissions_shifted)

Impulse = []
for t in range(len(time)):
    summa = 0
    for i in range(5):
        summa += A[i] * np.exp(-t/tau_all[t][i])
    Impulse.append(summa)
        summa += A[i] * np.exp(-t/tau_all[t][i])
    Impulse.append(summa)

plt.plot(time, Impulse)
plt.show()
"""

#TASK 4


def M(t):
    M = B_0[0]

    for t_tilde in range(0, t):
        M += I(t-t_tilde, t)*U[t_tilde]

    return M


plt.plot([M(t) for t in range(len(time))])
plt.show()



