from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model = ConcreteModel()

batteryOn = True

max_hydro = 33_000_000  #MWh


Co2_before = 138.75*1e6 # ton
Co2_cap = Co2_before * 0.1


# DATA
countries = ['DE', 'DK', 'SE']
techs = ['Wind', 'PV', 'Gas', 'Hydro', 'Battery']
tech_colors = ['gray', "orange", 'green', 'blue', 'purple']
eta = {'Wind' : 1, 'PV' : 1, 'Gas' : 0.4, 'Hydro' : 1, 'Battery' : 0.9} #you could also formulate eta as a time series with the capacity factors for PV and wind

transmission_lines = [('DESE'), ('DEDK'), ('DKSE'), ('SEDE'), ('DKDE'), ('SEDK')]


tech_data = {
    "Wind": {
        "inv_cost": 1_100_000,  #euro/MW
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 280_000, "DK": 90_000, "DE": 180_000}  #MW
    },
    "PV": {
        "inv_cost": 600_000,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 75_000, "DK": 60_000, "DE": 460_000}
    },
    "Gas": {
        "inv_cost": 550_000,
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
        "cap": {"SE": 14_000, "DK": 0, "DE": 0}
    },
    "Battery": {
        "inv_cost": 150_000,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 10,
        "eff": 0.9,
        "emission": 0,
        "cap": "inf"
    },
    "Transmission": {
        "inv_cost": 2_500_000,
        "run_cost": 0,
        "fuel_cost": 0,
        "lifetime": 50,
        "eff": 0.98,
        "emission": 0,
        "cap": "inf"
    },
    "Nuclear": {
        "inv_cost": 7_700_000,
        "run_cost": 4,
        "fuel_cost": 3.2,
        "lifetime": 50,
        "eff": 0.4,
        "emission": 0,
        "cap": "inf"
    }
}

discountrate = 0.05

input_data = pd.read_csv('data/TimeSeries.csv', header=[0], index_col=[0])


#TIME SERIES HANDLING
def demandData():
    demand = {}
    for n in model.nodes:
        for t in model.hours.data():
            demand[n,t] = float(input_data[f"Load_{n}"].iloc[t-1].item())
    return demand

def windData():
    wind = {}
    for n in model.nodes:
        for t in model.hours.data():
            wind[n,t] = float(input_data[f"Wind_{n}"].iloc[t-1].item())
    return wind

def solarData():
    solar = {}
    for n in model.nodes:
        for t in model.hours.data():
            solar[n,t] = float(input_data[f"PV_{n}"].iloc[t-1].item())
    return solar

def investmentData():
    inv = {}
    for g in model.gens:
        inv[g] = float(tech_data[g]['inv_cost'])
    return inv

def runningData():
    run = {}
    for g in model.gens:
        run[g] = float(tech_data[g]['run_cost'])
    return run

def fuelData():
    fuel = {}
    for g in model.gens:
        fuel[g] = float(tech_data[g]['fuel_cost'])
    return fuel

def lifetimeData():
    lifetime = {}
    for g in model.gens:
        lifetime[g] = float(tech_data[g]['lifetime'])
    return lifetime

def hydroInflowData():
    hydroInflowData = {}
    for t in model.hours.data():
        hydroInflowData[t] = float(input_data["Hydro_inflow"].iloc[t-1].item())
    return hydroInflowData


#SETS
model.nodes            = Set(initialize=countries         , doc='countries')
model.hours            = Set(initialize=input_data.index  , doc='hours')
model.gens             = Set(initialize=techs             , doc='techs')
model.transmisionlines = Set(initialize=transmission_lines, doc='transmision_lines')


#PARAMETERS
model.demand = Param(model.nodes, model.hours, initialize=demandData())
model.solar  = Param(model.nodes, model.hours, initialize=solarData())
model.wind   = Param(model.nodes, model.hours, initialize=windData())
model.hydro  = Param(model.hours, initialize=hydroInflowData())

model.eta = Param(model.gens, initialize=eta, doc='Conversion efficiency')


model.investment_cost = Param(model.gens, initialize=investmentData(), doc='Investments cost')
model.running_cost    = Param(model.gens, initialize=runningData()   , doc='Running costs')
model.fuel_cost       = Param(model.gens, initialize=fuelData()      , doc='Fuel costs')

model.lifetime = Param(model.gens, initialize=lifetimeData(), doc='liftime')


#VARIABLES
#capMaxdata = pd.read_csv('data/capMax.csv', index_col=[0])

def capacity_max(model, n, g):
    capMax = {}
    if tech_data[g]['cap'] !=  "inf":
        #capMax[n, g] = float(capMaxdata[g].loc[capMaxdata.index == n])
        capMax[n, g] = float(tech_data[g]['cap'][n])
        return 0.0, capMax[n,g]
    elif g == 'Battery' and not batteryOn:
        return 0.0, 0.0
    else:
        return 0.0, None

model.capa = Var(model.nodes, model.gens, bounds=capacity_max, doc='Generator cap')
model.prod = Var(model.nodes, model.gens, model.hours, doc='Generator prod')

model.hydroreservoir = Var(model.hours, doc='hydroreservoir')

model.batterystorage = Var(model.nodes, model.hours, doc="Currentt charge in batteries")

model.transmisions          = Var(model.transmisionlines, model.hours, bounds=(0, None), doc="Transmsions in the system")
model.transmisioncapacities = Var(model.transmisionlines, bounds=(0, None), doc='Generator cap')


#CONSTRAINTS
def prodcapa_rule(model, nodes, gens, time):
    return model.prod[nodes, gens, time] <= model.capa[nodes, gens]

def positive_rule(model, nodes, gens, time):
    return model.prod[nodes, gens, time] >= 0

def hydro_end_rule(model):
    return model.hydroreservoir[model.hours.first()] == model.hydroreservoir[model.hours.last()]

def hydro_max_rule(model, hours):
    return model.hydroreservoir[hours] <= max_hydro

def hydro_min_rule(model, hours):
    return model.hydroreservoir[hours] >= 0


def hydroflow_rule(model, h):
    if h != model.hours.last():  # Only apply if it's not the last hour
        return model.hydroreservoir[h+1] == model.hydroreservoir[h] + model.hydro[h+1] - model.prod["SE", "Hydro", h+1]
    else:
        return Constraint.Skip  # Skip constraint for the last hour



def battery_end_rule(model, nodes):
    return model.batterystorage[nodes, model.hours.first()] == model.batterystorage[nodes, model.hours.last()]

def battery_max_rule(model, hours, nodes):
    return model.batterystorage[nodes, hours] <= model.capa[nodes, "Battery"]

def battery_min_rule(model, hours, nodes):
    return model.batterystorage[nodes, hours] >= 0

def battery_flow_rule(model, hours, nodes):
    if hours != model.hours.last():  # Only apply if it's not the last hour
        net_transmisons = 0
        for line in model.transmisionlines:
            if line [0:2] == nodes:
                net_transmisons -= model.transmisions[line, hours]
            elif line [2:4] == nodes:
                net_transmisons += model.transmisions[line, hours]*0.98

        produced = sum([model.prod[nodes, g, hours] for g in model.gens]) + net_transmisons - model.prod[nodes, "Battery", hours]/0.9
        consumed = model.demand[nodes, hours]
        batterydiff = (produced - consumed)
        
        return model.batterystorage[nodes, hours+1] == model.batterystorage[nodes, hours] + batterydiff
    else:
        return Constraint.Skip  # Skip constraint for the last hour


def trans_max_rule(model, lines, hours):
    return model.transmisions[lines, hours] <= model.transmisioncapacities[lines]

def trans_sym_rule(model, t):
    for line in model.transmisionlines:
        if t[0:2] == line[2:4] and t[2:4] == line[0:2]:
            return model.transmisioncapacities[t] == model.transmisioncapacities[line]



def demand_satisfied_rule(model, n, t):
    net_transmisons = 0
    for line in model.transmisionlines:
        if line [0:2] == n:
            net_transmisons -= model.transmisions[line, t]
        elif line [2:4] == n:
            net_transmisons += model.transmisions[line, t]*0.98
    return sum(model.prod[n, g, t] for g in model.gens) + net_transmisons >= model.demand[n, t]


def windMax_rule(model, nodes, hours): # the wind can only blow so much
    return model.prod[nodes, 'Wind', hours] <= model.wind[nodes, hours] * model.capa[nodes, 'Wind']

def pvMax_rule(model, nodes, hours): # the sun can only shine so much
    return model.prod[nodes, 'PV', hours] <= model.solar[nodes, hours] * model.capa[nodes, 'PV']


def CO2_rule(model):
    CO2_tot = 0
    for n in model.nodes:
        for h in model.hours:
            CO2_tot += model.prod[n, "Gas", h]*0.202/0.4

    return CO2_tot <= Co2_cap

model.prodWind = Constraint(model.nodes, model.hours, rule=windMax_rule)
model.prodPV = Constraint(model.nodes, model.hours, rule=pvMax_rule)

model.hydroEnd = Constraint(rule=hydro_end_rule)
model.hydroMax = Constraint(model.hours, rule=hydro_max_rule)
model.hydroMin = Constraint(model.hours, rule=hydro_min_rule)
model.hydroFlow = Constraint(model.hours, rule=hydroflow_rule)


#BATTERY
model.batteryEnd  = Constraint(model.nodes, rule=battery_end_rule)
model.batteryMax  = Constraint(model.hours, model.nodes, rule=battery_max_rule)
model.batteryMin  = Constraint(model.hours, model.nodes, rule=battery_min_rule)
model.batteryFlow = Constraint(model.hours, model.nodes, rule=battery_flow_rule)

model.CO2Cap = Constraint(rule=CO2_rule)

model.demandSatisfied = Constraint(model.nodes, model.hours, rule=demand_satisfied_rule)
model.prodCapa = Constraint(model.nodes, model.gens, model.hours, rule=prodcapa_rule)
model.positive = Constraint(model.nodes, model.gens, model.hours, rule=positive_rule)

#TRANSMISIONLINES
model.transmisionsMax  = Constraint(model.transmisionlines, model.hours, rule=trans_max_rule)
model.transmisionsSym  = Constraint(model.transmisionlines, rule=trans_sym_rule)


#OBJECTIVE FUNCTION
def objective_rule(model):

    cost = 0
    anual_cost = 0
    running_cost = 0
    fuel_cost = 0 

    for n in model.nodes:
        for g in model.gens:
            discount_factor = discountrate / (1 - 1/((1 + discountrate)**model.lifetime[g]))
            anual_cost += model.capa[n, g] * discount_factor * model.investment_cost[g]

    for line in model.transmisionlines:
        discount_factor = discountrate / (1 - 1/((1 + discountrate)**50))
        anual_cost += model.transmisioncapacities[line] * discount_factor * 1_250_000


    for n in model.nodes:
        for g in model.gens:
            for h in model.hours:
                running_cost += model.prod[n,g,h] * model.running_cost[g]
                fuel_cost += model.prod[n,g,h] * model.fuel_cost[g] / model.eta[g]


    
    return anual_cost + running_cost + fuel_cost


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

    #Reading output - example
    capTot = {}
    for n in model.nodes:
        for g in model.gens:
            capTot[n, g] = model.capa[n, g].value/1e3 #GW


    CO2_tot = 0
    for n in model.nodes:
        for h in model.hours:
            CO2_tot += model.prod[n, "Gas", h].value*0.202/0.4

    
    costTot = value(model.objective) / 1e6 #Million EUR

    # PLOT 1 
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
    plt.title('Generator Production at Node DE Over hour 147 to 651')
    plt.xlabel('Hour')
    plt.ylabel('Production (MW)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("3_Production_DE_147-651.png", dpi = 300)
    plt.show()


    # PLOT 2

    energy_caps = {}
    for gen in techs:
        energy_caps[gen] = [model.capa[region, gen].value for region in countries]
    
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(countries))

    for boolean, weight_count in energy_caps.items():
        p = ax.bar(countries, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Capacity for each country")
    ax.legend(loc="upper right")

    plt.savefig("3_Capacity_per_tech.png", dpi = 300)

    plt.show()

    #PLOT 3
    energy_prod = {}
    for gen in techs:
        energy_prod[gen] = [sum(model.prod[region, gen, h].value for h in model.hours) for region in countries]

    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(countries))

    for boolean, weight_count in energy_prod.items():
        p = ax.bar(countries, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title("Total production for each country during the year")
    ax.legend(loc="upper right")

    plt.savefig("3_Production_per_tech.png", dpi = 300)
    plt.show()

    plt.bar(model.transmisionlines, [model.transmisioncapacities[line].value for line in model.transmisionlines])
    plt.savefig("3_transmision_capacities.png", dpi = 300)
    plt.title("Transmisioncapacities")
    plt.show()

    plt.bar(model.transmisionlines, [sum([model.transmisions[line, h].value for h in model.hours]) for line in model.transmisionlines])
    plt.savefig("3_transmision.png", dpi = 300)
    plt.title("Transmisions")
    plt.show()

    print("Total system cost (Million EUR): ", costTot)
    print("Co2 emmision: ", CO2_tot)