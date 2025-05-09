from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

model = ConcreteModel()


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


#PARAMETERS
model.demand = Param(model.nodes, model.hours, initialize=demandData())
model.eta = Param(model.gens, initialize=eta, doc='Conversion efficiency')

model.wind = Param(model.nodes, model.hours, initialize=windData())
model.solar = Param(model.nodes, model.hours, initialize=solarData())

model.investment_cost = Param(model.gens, initialize=investmentData())
model.running_cost = Param(model.gens, initialize=runningData())
model.fuel_cost = Param(model.gens, initialize=fuelData())

model.lifetime = Param(model.gens, initialize=lifeData())


#VARIABLES
#capMaxdata = pd.read_csv('data/capMax.csv', index_col=[0])

batteryOn = False

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


#CONSTRAINTS
def prodcapa_rule(model, nodes, gens):
    sum = 0
    for h in model.hours:
        sum += model.prod[nodes, gens, h]
    return sum <= model.capa[nodes, gens]

def prodDemand_rule(model, nodes, hours):
    sum = 0
    for g in model.gens:
        sum += model.prod[nodes, g, hours]
    return sum >= model.demand[nodes, hours]

def windMax_rule(model, nodes, hours): # the wind can only blow so much
    return model.prod[nodes, 'Wind', hours] <= model.wind[nodes, hours] * model.capa[nodes, 'Wind']

def pvMax_rule(model, nodes, hours): # the sun can only shine so much
    return model.prod[nodes, 'PV', hours] <= model.solar[nodes, hours] * model.capa[nodes, 'PV']

def capaMin_rule(model, nodes, gens):
    return model.capa[nodes, gens] >= 0

def prodMin_rule(model, nodes, gens, hours):
    return model.prod[nodes, gens, hours] >= 0

model.prodCapa = Constraint(model.nodes, model.gens, rule=prodcapa_rule)
model.prodDemand = Constraint(model.nodes, model.hours, rule=prodDemand_rule)
model.prodWind = Constraint(model.nodes, model.hours, rule=windMax_rule)
model.prodPV = Constraint(model.nodes, model.hours, rule=pvMax_rule)
model.capaMin = Constraint(model.nodes, model.gens, rule=capaMin_rule)
model.prodMin = Constraint(model.nodes, model.gens, model.hours, rule=prodMin_rule)


#OBJECTIVE FUNCTION
def objective_rule(model):

    cost = 0
    annual_cost = 0
    running_cost = 0
    fuel_cost = 0

    for n in model.nodes:
        for g in model.gens:
            discount_factor = discountrate/(1-1/(1+discountrate)**model.lifetime[g])
            annual_cost += model.capa[n,g] * model.investment_cost[g]*discount_factor

    for n in model.nodes:
        for g in model.gens:
            for h in model.hours:
                running_cost += model.prod[n,g,h] * model.running_cost[g]

    return annual_cost + running_cost

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
