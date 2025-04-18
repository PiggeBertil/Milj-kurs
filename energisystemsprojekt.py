from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

model = ConcreteModel()

batteryOn = False

# DATA
countries = ['DE', 'DK', 'SE']
techs = ['Wind', 'PV', 'Gas', 'Hydro', 'Battery']
eta = {'Wind' : 1, 'PV' : 1, 'Gas' : 0.4, 'Hydro' : 1, 'Battery' : 0.9} #you could also formulate eta as a time series with the capacity factors for PV and wind



tech_data = {
    "Wind": {
        "inv_cost": 1100,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 280, "DK": 90, "DE": 180}
    },
    "PV": {
        "inv_cost": 600,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 25,
        "eff": None,
        "emission": 0,
        "cap": {"SE": 75, "DK": 60, "DE": 460}
    },
    "Gas": {
        "inv_cost": 550,
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
        "cap": {"SE": 14, "DK": 0, "DE": 0}
    },
    "Battery": {
        "inv_cost": 150,
        "run_cost": 0.1,
        "fuel_cost": 0,
        "lifetime": 10,
        "eff": 0.9,
        "emission": 0,
        "cap": "inf"
    },
    "Transmission": {
        "inv_cost": 2500,
        "run_cost": 0,
        "fuel_cost": 0,
        "lifetime": 50,
        "eff": 0.98,
        "emission": 0,
        "cap": "inf"
    },
    "Nuclear": {
        "inv_cost": 7700,
        "run_cost": 4,
        "fuel_cost": 3.2,
        "lifetime": 50,
        "eff": 0.4,
        "emission": 0,
        "cap": "inf"
    }
}

#tech_data_df = pd.DataFrame.from_dict(tech_data, orient='index')


discountrate = 0.05

input_data = pd.read_csv('data/TimeSeries.csv', header=[0,1], index_col=[0])


#TIME SERIES HANDLING
def demandData():
    demand = {}
    for n in model.nodes:
        for t in model.hours.data():
            demand[n,t] = float(input_data[f"Load_{n}"].values[t-1])
    return demand

def windData():
    wind = {}
    for n in model.nodes:
        for t in model.hours.data():
            wind[n,t] = float(input_data[f"Wind_{n}"].values[t-1])
    return wind

def solarData():
    solar = {}
    for n in model.nodes:
        for t in model.hours.data():
            solar[n,t] = float(input_data[f"PV_{n}"].values[t-1])
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


#SETS
model.nodes = Set(initialize=countries, doc='countries')
model.hours = Set(initialize=input_data.index, doc='hours')
model.gens  = Set(initialize=techs, doc='techs')


#PARAMETERS
model.demand = Param(model.nodes, model.hours, initialize=demandData())
model.solar = Param(model.nodes, model.hours, initialize=solarData())
model.wind = Param(model.nodes, model.hours, initialize=windData())

model.eta = Param(model.gens, initialize=eta, doc='Conversion efficiency')


model.investment_cost = Param(model.gens, initialize=investmentData(), doc='Investments cost')
model.running_cost = Param(model.gens, initialize=runningData(), doc='Running costs')
model.fuel_cost = Param(model.gens, initialize=fuelData(), doc='Fuel costs')

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
model.prod = Var(model.nodes, model.gens, model.hours, doc='Generator cap')


#CONSTRAINTS
def prodcapa_rule(model, nodes, gens, time):
    return model.prod[nodes, gens, time] <= model.capa[nodes, gens]


model.prodCapa = Constraint(model.nodes, model.gens, model.hours, rule=prodcapa_rule)


#OBJECTIVE FUNCTION
def objective_rule(model):

    cost = 0
    anual_cost = 0

    for n in model.nodes:
        for g in model.gens:
            discount_factor = discountrate / (1 - 1/((1 + discountrate)**model.lifetime[g]))
            anual_cost += model.capa[n, g] * discount_factor * model.investment_cost[g]


    
    return anual_cost

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


    costTot = value(model.objective) / 1e6 #Million EUR