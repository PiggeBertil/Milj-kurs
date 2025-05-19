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


F_0 = np.zeros((3, 3))

NPP_0 = 60.0

F_0[1,0] = 15.0
F_0[1,2] = 45.0
F_0[2,0] = 45.0
F_0[0,1] = NPP_0

F = F_0

alpha = F_0[:, :] / B_0[:, np.newaxis]

def simulate(BETA = 0.3, time = time):


    Bs = [B_0] # GtC

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


plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "])
plt.plot(time, [M(t)*Conversion_factor for t in range(len(time))])
plt.show()

#Task 5

BETA = 0.25

Bs = simulate(BETA=BETA)


def M_combinded(t):

    M = B_0[0]

    for t_tilde in range(0, t):
        B = Bs[t_tilde]

        NPP = NPP_0 * (1 + BETA*np.log(B[0]/B_0[0]))


        U_new = alpha[2,0]*B[2] + alpha[1,0]*B[1] - NPP + U[t_tilde]
        M += I(t-t_tilde, t)*U_new

    return M





plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "])
plt.plot(time, [M_combinded(t)*Conversion_factor for t in range(len(time))], label= f"B_1 Beta={BETA}", color = "red")
plt.plot(time, [B[1]*Conversion_factor for B in Bs], label= f"B_2 Beta={BETA}", color = "blue")
plt.plot(time, [B[2]*Conversion_factor for B in Bs], label= f"B_3 Beta={BETA}", color = "green")


plt.plot(time, [(Bs[t][0] - M_combinded(t))*Conversion_factor for t in range(len(time))], label = "B_4 Beta=0.25", color = "orange")

plt.legend()

plt.show()


#Task 8

p_co2 = concentrations_df["CO2ConcRCP45 (ppm CO2) "].values
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)

filename = 'data/radiativeForcingRCP45.csv'
radiativeForcing_df = pd.read_csv(filename)

plt.plot(time, RF_co2, color='red')
plt.plot(time, radiativeForcing_df["RF CO2 (W/m2)"], color='blue')
plt.show()

# Task 9

s=1
RF_sum = radiativeForcing_df["RF CO2 (W/m2)"].values + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

plt.plot(time, RF_sum)
plt.show()

# Task 10

# Constants
c = 4186  # specific heat capacity of water [J/(kg·K)]
rho = 1020  # density of water [kg/m^3]
h = 50  # effective depth of surface box [m]
d = 2000  # effective depth of deep ocean box [m]

# Parameters with uncertainty spans as comments
lambda_climate = 0.5  # climate sensitivity parameter [K·W^-1·m^2]
# Uncertainty span: 0.5 - 1.3 [K·W^-1·m^2]

kappa = 1  # heat exchange coefficient [W·K^-1·m^-2]
# Uncertainty span: 0.2 - 1 [W·K^-1·m^-2]

# Radiative forcing (RF) is just defined as a parameter in W/m^2, value not specified
RF = np.ones(len(time))  # radiative forcing [W/m^2] - value to be defined

# Effective heat capacities
C1 = c * h * rho  # surface box heat capacity [J/(m^2·K)]
C2 = c * d * rho  # deep ocean box heat capacity [J/(m^2·K)]

# Convert to [W·yr·K^-1·m^-2] by converting J to W·yr
seconds_per_year = 365.25 * 24 * 60 * 60  # [s/yr]
C1 /= seconds_per_year  # [W·yr·K^-1·m^-2]
C2 /= seconds_per_year  # [W·yr·K^-1·m^-2]

def temperature_response(lambda_climate = lambda_climate, kappa=kappa):
    deltaT_0 = np.array([0,0])
    deltaTs = [deltaT_0]

    for t in range(len(time))[1:]:
        deltaT = deltaTs[-1]

        dT = np.zeros(2)

        dT[0] = (RF[t] - deltaT[0]/lambda_climate - kappa*(deltaT[0]-deltaT[1]))/C1
        dT[1] = (kappa * (deltaT[0]-deltaT[1]))/C2

        deltaTs.append(deltaT + dT)
    
    return deltaTs

deltaTs = temperature_response(lambda_climate, kappa)
plt.plot(time, [deltaTs[t][0] for t in range(len(time))], label = "delta T1")
plt.plot(time, [deltaTs[t][1] for t in range(len(time))], label = "delta T2")
plt.legend()
plt.show()

# Task 10 b)

def e_fold(delta_Ts, T_final):
    for i in range(len(delta_Ts)):
        if delta_Ts[i] > (1-np.exp(-1))*T_final:
            return i
    return "för långt"

print("För T1: ", e_fold([deltaTs[t][0] for t in range(len(time))], RF[-1] * lambda_climate))
print("För T2: ", e_fold([deltaTs[t][1] for t in range(len(time))], RF[-1] * lambda_climate))