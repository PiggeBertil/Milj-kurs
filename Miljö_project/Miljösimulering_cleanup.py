import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename = 'data/koncentrationerRCP45.csv'
concentrations_df = pd.read_csv(filename)

filename = 'data/utslappRCP45.csv'
emisions_df = pd.read_csv(filename)

time = concentrations_df["Time (year)"].values
U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])



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

def simulate(BETA = 0.35, time = time):


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

tau_0 = np.array([2.0, 12.2, 50.4, 243.3, np.inf])

def tau(t, i, U):
    k = 3.06 * 10 ** (-3)
    tau_0 = np.array([2.0, 12.2, 50.4, 243.3, np.inf])

    return tau_0[i] * (1 + k*sum(U[:t]))

def I(t, t_prim, U):
    A = np.array([0.113, 0.213, 0.258, 0.273, 0.1430])
    I_sum = sum([A[i]*np.exp(-t/tau(t_prim, i, U)) for i in range(5)])
    return I_sum



plt.plot([I(t, 100, U) for t in range(500)], label = "Impulse of carbon in year 100")
plt.plot([I(t, 150, U) for t in range(500)], label = "Impulse of carbon in year 150")
plt.plot([I(t, 200, U) for t in range(500)], label = "Impulse of carbon in year 200")
plt.plot([I(t, 250, U) for t in range(500)], label = "Impulse of carbon in year 250")
plt.plot([I(t, 300, U) for t in range(500)], label = "Impulse of carbon in year 300")
plt.legend()
plt.show()

#Task 4

def M(t, U):
    M = B_0[0]

    for t_tilde in range(0, t):
        M += I(t-t_tilde, t, U)*U[t_tilde]

    return M


plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "])
plt.plot(time, [M(t, U)*Conversion_factor for t in range(len(time))])
plt.show()



#Task 5-7

#Kommenterad för den tod sån jävla tid att köra ersätter med nedan 

Bs = [[M(t, U)*Conversion_factor, 0, 0] for t in range(len(time))]

#Bs = [B_0]
#U_news = []
#
#BETA = 0.35
#
#B4s = [0]
#
#
#for t in range(1, len(time)):
#    B = Bs[-1]
#
#    NPP = NPP_0 * (1 + BETA*np.log(B[0]/B_0[0]))
#
#    dB = np.zeros(3)
#
#    U_news.append(alpha[2,0]*B[2] + alpha[1,0]*B[1] - NPP + U[t])
#
#
#    B4s.append(B[0] + U_news[-1] - M(t, U_news))
#
#    B[0]  = M(t, U_news)
#    dB[1] = NPP - alpha[1,2]*B[1] - alpha[1,0]*B[1]
#    dB[2] = alpha[1,2]*B[1] - alpha[2,0]*B[2]
#
#
#    Bs.append(B + dB)
#
#
#
#
#plt.plot(time, concentrations_df["CO2ConcRCP45 (ppm CO2) "])
#plt.plot(time, [B[0]*Conversion_factor for B in Bs ],                            label= f"B_1 Beta={BETA}", color = "red")
#plt.plot(time, [B[1]*Conversion_factor for B in Bs ],                            label= f"B_2 Beta={BETA}", color = "blue")
#plt.plot(time, [B[2]*Conversion_factor for B in Bs ],                            label= f"B_3 Beta={BETA}", color = "green")
#plt.plot(time, [B*Conversion_factor    for B in B4s],                            label =f"B_4 Beta={BETA}", color = "orange")
#
#plt.xlim((1765, 2100))
#
#plt.legend()
#plt.show()


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

# Task 10 a)

# Constants
c = 4186  # specific heat capacity of water [J/(kg·K)]
rho = 1020  # density of water [kg/m^3]
h = 50  # effective depth of surface box [m]
d = 2000  # effective depth of deep ocean box [m]

# Parameters with uncertainty spans as comments
lambda_climate = 0.5  # climate sensitivity parameter [K·W^-1·m^2]
# Uncertainty span: 0.5 - 1.3 [K·W^-1·m^2]

kappa = 0.5  # heat exchange coefficient [W·K^-1·m^-2]
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

def temperature_response(lambda_climate = lambda_climate, kappa=kappa, RF=RF):
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



#Task 11 a)

BETA = 0.35
kappa = 0.5


filename = 'data/NASA_GISS.csv'
NASA_df = pd.read_csv(filename)

NASA_time     = NASA_df["Year"].values
NASA_tempdiff = NASA_df["No_Smoothing"].values


Ms = [B[0]*Conversion_factor for B in Bs ]

p_co2 = np.array(Ms)
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)


RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(lambda_climate, kappa, RF = RF_sum)

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

shift= np.mean(T1s[1951-1765:1980-1765])

plt.plot(time, np.array(T1s)-shift, label = "delta T1")
plt.plot(time, np.array(T2s)-shift, label = "delta T2")

plt.plot(NASA_time, NASA_tempdiff, linestyle = "dashed", label = "NASA temp_diff")

plt.xlim((1765, 2024))
plt.title("Simulerad och mätt temperatur över tid")
plt.legend()
plt.show()


#11 b)

############################################################################################################################################
#This is for optimizing

#minimum = 10000000
#for s in [0.1, 0.2, 0.4, 0.8, 1, 1.2, 1.4, 1.6, 2, 2.5]:
#    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
#
#        RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values
#
#        deltaTs = temperature_response(1.3, k, RF = RF_sum) 
#
#        T1s =  [deltaTs[t][0] for t in range(len(time))]
#        T2s =  [deltaTs[t][1] for t in range(len(time))]
#
#        shift= np.mean(T1s[1951-1765:1980-1765])
#
#        if np.abs(np.array(T1s[1880-1765:2025-1765]) - shift - NASA_tempdiff).sum() < minimum:
#            minimum = np.abs(np.array(T1s[1880-1765:2025-1765]) - shift - NASA_tempdiff).sum()
#
#            print(s, k, minimum)
        


############################################################################################################################################
#lamda = 0.5, s=0.8, kappa = 0.2
s=0.8
kappa = 0.2
RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(0.5, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

shift= np.mean(T1s[1951-1765:1980-1765])

plt.plot(time, np.array(T1s)-shift, label = "delta T1, lambda = 0.5")

############################################################################################################################################
#lamda = 0.8, s=1, kappa=0.7
s=1
kappa = 0.7
RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(0.8, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

shift= np.mean(T1s[1951-1765:1980-1765])

plt.plot(time, np.array(T1s)-shift, label = "delta T1, lambda = 0.8")
############################################################################################################################################
#lamda = 1.3, s=1.2, kappa=1
s=1.2
kappa = 1

RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(1.3, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

shift= np.mean(T1s[1951-1765:1980-1765])

plt.plot(time, np.array(T1s)-shift, label = "delta T1, lamda = 1.3")
############################################################################################################################################

plt.plot(NASA_time, NASA_tempdiff, label = "NASA delta T1")

plt.xlim((1880, 2024))
plt.legend()
plt.show()



#Task 12


LAMBDA = 0.8
s=1
kappa = 0.7
##############################################################################################################################

U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])

U[2024-1765:2070-1765] = np.arange(U[2024-1765], 0, -U[2024-1765]/(2070-2024))
U[2070-1765:] = 0


Ms = [M(t, U)*Conversion_factor for t in range(len(time))]

p_co2 = np.array(Ms)
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)

RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(LAMBDA, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

plt.plot(time, np.array(T1s), label = "delta T1, lambda = 0.8, case 1")

print(f"For case 1 the temperature increased: {T1s[2100-1765]}")

##############################################################################################################################

U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])

U[2024-1765:] = U[2024-1765]

Ms = [M(t, U)*Conversion_factor for t in range(len(time))]

p_co2 = np.array(Ms)
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)

RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(LAMBDA, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

plt.plot(time, np.array(T1s), label = "delta T1, lambda = 0.8, case 2")

print(f"For case 2 the temperature increased: {T1s[2100-1765]}")

##############################################################################################################################

U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])

U[2024-1765:2100-1765] = np.arange(U[2024-1765], 2*U[2024-1765], U[2024-1765]/(2100-2024))
U[2100-1765:] = 2*U[2024-1765]



Ms = [M(t, U)*Conversion_factor for t in range(len(time))]

p_co2 = np.array(Ms)
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)

RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

deltaTs = temperature_response(LAMBDA, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

plt.plot(time, np.array(T1s), label = "delta T1, lambda = 0.8, case 3")

print(f"For case 3 the temperature increased: {T1s[2100-1765]}")

##############################################################################################################################

plt.xlim((1765, 2200))
plt.legend()
plt.title("A plot showing the medium surface temperature over time for different CO2-emmision scenarios")
plt.show()




#Task 13

plt.plot(time, np.array(T1s), label = "delta T1, lambda = 0.8, case 3, without areos")

U = np.array([emisions_df.loc[emisions_df['Time (year)'] == t].values[0][1] for t in time])

U[2024-1765:2100-1765] = np.arange(U[2024-1765], 2*U[2024-1765], U[2024-1765]/(2100-2024))
U[2100-1765:] = 2*U[2024-1765]



Ms = [M(t, U)*Conversion_factor for t in range(len(time))]

p_co2 = np.array(Ms)
p_co2_0 = p_co2[0]

RF_co2 = 5.35 * np.log(p_co2/p_co2_0)

RF_sum = RF_co2 + radiativeForcing_df["RF aerosols (W/m2)"].values * s + radiativeForcing_df["RF other than CO2 and aerosols (W/m2)"].values

RF_sum[2050-1765:2100-1765] = RF_sum[2050-1765:2100-1765] - 4

deltaTs = temperature_response(LAMBDA, kappa, RF = RF_sum) 

T1s =  [deltaTs[t][0] for t in range(len(time))]
T2s =  [deltaTs[t][1] for t in range(len(time))]

plt.plot(time, np.array(T1s), label = "delta T1, lambda = 0.8, case 3 with areosol")

print(T1s[2100-1765])


plt.xlim((1880, 2200))
plt.legend()
plt.title("A plot showing the effekt of aereosol")
plt.show()