# exercise 2 energisystem

from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model = ConcreteModel()

batteryOn = True

max_hydro = 33_000_000 # MWh


Co2_before = 144194510.29937696 # ton
Co2_cap = Co2_before * 0.1

# DATA
countries = ['DE', 'DK', 'SE']
techs = ['Wind', 'PV', 'Gas', 'Hydro', 'Battery']
eta = {'Wind' : 1, 'PV' : 1, 'Gas' : 0.4, 'Hydro' : 1, 'Battery' : 0.9} #you could also formulate eta as a time series with the capacity factors for PV and wind

transmission_lines = [('DESE'), ('DEDK'), ('DKSE'), ('SEDE'), ('DKDE'), ('SEDK')]


tech_data = {
    "Wind": {
        "inv_cost": 1_100_000, #MW
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 280_000, "DK": 90_000, "DE": 180_000} # MW
    },
    "PV": {
        "inv_cost": 600_000, #MW
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 75_000, "DK": 60_000, "DE": 460_000} # MW
    },
    "Gas": {
        "inv_cost": 550_000, #MW
        "run_cost": 2,
        "fuel_cost": 22,
        "lifetime": 30,
        "eff": 0.4,
        "emission": 0.202,
        "cap": "inf"
    },
    "Hydro": {
        "inv_cost": 0,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 80,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 14_000, "DK": 0, "DE": 0} # MW
    },
    "Battery": {
        "inv_cost": 150_000, #MW
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 10,
        "eff": 0.9,
        "emission": 0,
        "cap": "inf"
    },
    "Transmission": {
        "inv_cost": 2_500_000, #MW
        "run_cost": 0,
        "fuel_cost": 0,
        "lifetime": 50,
        "eff": 0.98,
        "emission": 0,
        "cap": "inf"
    },
    "Nuclear": {
        "inv_cost": 7_700_000, #MW
        "run_cost": 4,
        "fuel_cost": 3.2,
        "lifetime": 50,
        "eff": 0.4,
        "emission": 0,
        "cap": "inf"
    }
}

discountrate = 0.05

input_data = pd.read_csv('data/TimeSeries.csv', header=[0,1], index_col=[0])


#TIME SERIES HANDLING
def demandData():
    demand = {}
    for n in model.nodes:
        for t in model.hours:
            demand[n,t] = float(input_data[f"Load_{n}"].iloc[t-1].item())
    return demand

def windData():
    wind = {}
    for n in model.nodes:
        for t in model.hours:
            wind[n,t] = float(input_data[f"Wind_{n}"].iloc[t-1].item())
    return wind

def solarData():
    solar = {}
    for n in model.nodes:
        for t in model.hours:
            solar[n,t] = float(input_data[f"PV_{n}"].iloc[t-1].item())
    return solar

def hydroInflowData():
    hydro = {}
    for t in model.hours:
        hydro[t] = float(input_data["Hydro_inflow"].iloc[t-1].item())
    return hydro

def investmentData():
    inv = {}
    for g in model.gens:
        inv[g] = float(tech_data[g]["inv_cost"])
    return inv

def runningData():
    run = {}
    for g in model.gens:
        run[g] = float(tech_data[g]["run_cost"])
    return run

def fuelData():
    fuel = {}
    for g in model.gens:
        fuel[g] = float(tech_data[g]["fuel_cost"])
    return fuel

def lifeData():
    life = {}
    for g in model.gens:
        life[g] = float(tech_data[g]["lifetime"])
    return life


#SETS
model.nodes = Set(initialize=countries, doc='countries')
model.hours = Set(initialize=input_data.index, doc='hours')
model.gens = Set(initialize=techs, doc='techs')
model.transmisionlines = Set(initialize=transmission_lines, doc='transmision_lines')


#PARAMETERS
model.demand = Param(model.nodes, model.hours, initialize=demandData())
model.eta = Param(model.gens, initialize=eta, doc='Conversion efficiency')

model.wind = Param(model.nodes, model.hours, initialize=windData())
model.solar = Param(model.nodes, model.hours, initialize=solarData())
model.hydro = Param(model.hours, initialize=hydroInflowData())

model.investment_cost = Param(model.gens, initialize=investmentData())
model.running_cost = Param(model.gens, initialize=runningData())
model.fuel_cost = Param(model.gens, initialize=fuelData())

model.lifetime = Param(model.gens, initialize=lifeData())


#VARIABLES
#capMaxdata = pd.read_csv('data/capMax.csv', index_col=[0])


def capacity_max(model, n, g):
    capMax = {}
    if tech_data[g]['cap'] != "inf":
        capMax[n, g] = float(tech_data[g]['cap'][n])
        return 0.0, capMax[n,g]
    elif g == 'Battery' and not batteryOn:
        return 0.0, 0.0
    else:
        return 0.0, None

model.capa = Var(model.nodes, model.gens, bounds=capacity_max, doc='Generator cap')
model.prod = Var(model.nodes, model.gens, model.hours, doc='Generator prod')
model.hydroreservoir = Var(model.hours, doc="Hydro Reservoir")

model.batterystorage = Var(model.nodes, model.hours, doc="Currentt charge in batteries")

model.transmisions = Var(model.transmisionlines, model.hours, doc="Transmsions in the system")
model.transmisioncapacities = Var(model.transmisionlines, bounds=(0, None), doc='Generator cap')



#CONSTRAINTS
def prodcapa_rule(model, nodes, gens, hours):
    return model.prod[nodes, gens, hours] <= model.capa[nodes, gens]

def prodDemand_rule(model, nodes, hours):
    sum = 0
    for g in model.gens:
        sum += model.prod[nodes, g, hours]

    for t in model.transmisionlines:
        if t[1] == nodes:
            sum += model.transmisions[t, hours]*0.98
        elif t[0] == nodes:
            sum -= model.transmisions[t, hours]


    return sum >= model.demand[nodes, hours]

def windMax_rule(model, nodes, hours): # the wind can only blow so much
    return model.prod[nodes, 'Wind', hours] <= model.wind[nodes, hours] * model.capa[nodes, 'Wind']

def pvMax_rule(model, nodes, hours): # the sun can only shine so much
    return model.prod[nodes, 'PV', hours] <= model.solar[nodes, hours] * model.capa[nodes, 'PV']



def hydro_end_rule(model):
    return model.hydroreservoir[1] == model.hydroreservoir[8759]

def hydro_max_rule(model, hours):
    return model.hydroreservoir[hours] <= max_hydro

def hydro_min_rule(model, hours):
    return model.hydroreservoir[hours] >= 0

def hydroflow_rule(model, h):
    if h != model.hours.last():  # Only apply if it's not the last hour
        return model.hydroreservoir[h+1] == model.hydroreservoir[h] + model.hydro[h] - model.prod["SE", "Hydro", h]
    else:
        return Constraint.Skip  # Skip constraint for the last hour



def battery_end_rule(model, nodes):
    return model.batterystorage[nodes, 1] == model.batterystorage[nodes, 8759]

def battery_max_rule(model, hours, nodes):
    return model.batterystorage[nodes, hours] <= model.capa[nodes, "Battery"]

def battery_min_rule(model, hours, nodes):
    return model.batterystorage[nodes, hours] >= 0

def battery_flow_rule(model, hours, nodes):
    if hours != model.hours.last():  # Only apply if it's not the last hour
        produced = sum([model.prod[nodes, g, hours] for g in model.gens]) - model.prod[nodes, "Battery", hours]
        consumed = model.demand[nodes, hours]
        batterydiff = produced - consumed
        return model.batterystorage[nodes, hours+1] == model.batterystorage[nodes, hours] + batterydiff
    else:
        return Constraint.Skip  # Skip constraint for the last hour


def transmision_max(model, hours, transmisionlines):
    return model.transmisions[transmisionlines, hours] <= model.transmisioncapacities[transmisionlines]

def transmision_min(model, hours, transmisionlines):
    return model.transmisions[transmisionlines, hours] >= 0

def transmision_caps(model, t):
    for line in model.transmisionlines:
        if t[0:1] == line[2:3] and t[2:3] == line[0:1]:
            return model.transmisioncapacities[t] == model.transmisioncapacities[line]





def capaMin_rule(model, nodes, gens):
    return model.capa[nodes, gens] >= 0

def prodMin_rule(model, nodes, gens, hours):
    return model.prod[nodes, gens, hours] >= 0

# NEW RULE
def co2Max_rule(model):
    CO2 = 0
    for node in model.nodes:
        for hour in model.hours:
            CO2 += model.prod[node, "Gas", hour]
    return CO2 <= Co2_cap


model.prodCapa = Constraint(model.nodes, model.gens, model.hours, rule=prodcapa_rule)
model.prodDemand = Constraint(model.nodes, model.hours, rule=prodDemand_rule)
model.prodWind = Constraint(model.nodes, model.hours, rule=windMax_rule)
model.prodPV = Constraint(model.nodes, model.hours, rule=pvMax_rule)


#HYDRO
model.hydroEnd = Constraint(rule=hydro_end_rule)
model.hydroMax = Constraint(model.hours, rule=hydro_max_rule)
model.hydroMin = Constraint(model.hours, rule=hydro_min_rule)
model.hydroFlow = Constraint(model.hours, rule=hydroflow_rule)

#BATTERY
model.batteryEnd  = Constraint(model.nodes, rule=battery_end_rule)
model.batteryMax  = Constraint(model.hours, model.nodes, rule=battery_max_rule)
model.batteryMin  = Constraint(model.hours, model.nodes, rule=battery_min_rule)
model.batteryFlow = Constraint(model.hours, model.nodes, rule=battery_flow_rule)

#TRANSMISIONLINES
model.transmisionsMax  = Constraint(model.hours, model.transmisionlines, rule=transmision_max)
model.transmisionsMin  = Constraint(model.hours, model.transmisionlines, rule=transmision_min)
model.transmisionsCap = Constraint(model.transmisionlines, rule=transmision_caps)

#CAPACITY PRODUCTION
model.capaMin = Constraint(model.nodes, model.gens, rule=capaMin_rule)
model.prodMin = Constraint(model.nodes, model.gens, model.hours, rule=prodMin_rule)

# NEW CONSTRAINT (CO2)
model.co2Max = Constraint(rule = co2Max_rule)


#OBJECTIVE FUNCTION
def objective_rule(model):

    annual_cost = 0
    running_cost = 0
    fuel_cost = 0

    for n in model.nodes:
        for g in model.gens:
            discount_factor = discountrate/(1-1/(1+discountrate)**model.lifetime[g])
            annual_cost += model.capa[n,g] * model.investment_cost[g]*discount_factor

    for n in model.transmisionlines:
        discount_factor = discountrate/(1-1/(1+discountrate)**50)
        annual_cost += model.transmisioncapacities[n] * 1250 * discount_factor

    for n in model.nodes:
        for g in model.gens:
            for h in model.hours:
                running_cost += model.prod[n,g,h] * model.running_cost[g]
                fuel_cost += model.prod[n,g,h] * model.fuel_cost[g] / model.eta[g]

    return annual_cost + running_cost + fuel_cost

model.objective = Objective(rule=objective_rule, sense=minimize, doc='Objective function')


if __name__ == '__main__':
    from pyomo.opt import SolverFactory
    import pyomo.environ
    import pandas as pd

    opt = SolverFactory("gurobi_direct")
    opt.options["threads"] = 4
    print('Solving')
    results = opt.solve(model, tee=True)

    results.write()

    CO2_tot = 0
    for n in model.nodes:
        for h in model.hours:
            CO2_tot += model.prod[n, "Gas", h].value*0.202/0.4


    costTot = value(model.objective) / 1e6 #Million EUR


    # Plot of production in Germany with load

    energy_plots = []
    for gen in techs:
        energy_plot = [model.prod["DE", gen, h].value for h in range(147, 651)]
        energy_plots.append(energy_plot)

    # X-axis: hours
    hours = range(147, 651)
    demand = [model.demand["DE", i] for i in range(147, 651)]


    # Create the stackplot
    plt.figure(figsize=(12, 6))
    plt.stackplot(hours, energy_plots, labels=techs)

    plt.plot(hours, demand, label = "demand", color="black")

    # Customize the plot
    plt.title('Generator Production at Node DE Over three weeks in January')
    plt.xlabel('Hour')
    plt.ylabel('Production (MW)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("4_Production_DE_147-651.png", dpi = 300)
    plt.show()

    # Plot of installed capacities

    energy_caps = {}
    for gen in techs:
        energy_caps[gen] = [model.capa[region, gen].value for region in countries]

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(countries))

    for boolean, weight_count in energy_caps.items():
        p = ax.bar(countries, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Capacity")
    ax.legend(loc="upper right")

    plt.savefig("4_Capacity_per_tech.png", dpi = 300)
    plt.show()

    # Plot of total annual production

    energy_prod = {}
    for gen in techs:
        energy_prod[gen] = [sum(model.prod[region, gen, h].value for h in model.hours) for region in countries]

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(countries))

    for boolean, weight_count in energy_prod.items():
        p = ax.bar(countries, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Production")
    ax.legend(loc="upper right")

    plt.savefig("4_Production_per_tech.png", dpi = 300)
    plt.show()

    print("Total system cost (Million EUR): ", costTot)
    print("Co2 emmision: ", CO2_tot)