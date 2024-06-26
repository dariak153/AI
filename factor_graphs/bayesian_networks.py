from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


car_model_updated = BayesianModel([
    ('Battery', 'Radio'),
    ('Battery', 'Ignition'),
    ('Battery', 'StarterMotor'),  # Battery as a parent
    ('Ignition', 'Starts'),
    ('Gas', 'Starts'),
    ('StarterMotor', 'Starts'),
    ('NotIcyWeather', 'Starts'),  # NotIcyWeather as a parent
    ('Starts', 'Moves')
])


cpd_battery = TabularCPD('Battery', 2, [[0.7], [0.3]], state_names={'Battery': ['True', 'False']})
cpd_radio = TabularCPD('Radio', 2, [[0.9, 0], [0.1, 1]], evidence=['Battery'], evidence_card=[2], state_names={'Radio': ['True', 'False'], 'Battery': ['True', 'False']})
cpd_ignition = TabularCPD('Ignition', 2, [[0.97, 0], [0.03, 1]], evidence=['Battery'], evidence_card=[2], state_names={'Ignition': ['True', 'False'], 'Battery': ['True', 'False']})
cpd_gas = TabularCPD('Gas', 2, [[0.5], [0.5]], state_names={'Gas': ['True', 'False']})
cpd_not_icy_weather = TabularCPD('NotIcyWeather', 2, [[0.9], [0.1]], state_names={'NotIcyWeather': ['True', 'False']})
cpd_starter_motor = TabularCPD('StarterMotor', 2, [[0.95, 0.05], [0.05, 0.95]], evidence=['Battery'], evidence_card=[2], state_names={'StarterMotor': ['True', 'False'], 'Battery': ['True', 'False']})
cpd_starts = TabularCPD('Starts', 2,
                        [[0.85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0.15, 1, 1, 1, 1, 1, 1, 1, 1,1 , 1, 1, 1, 1, 1, 1]],
                        evidence=['Ignition', 'Gas', 'StarterMotor', 'NotIcyWeather'],
                        evidence_card=[2, 2, 2, 2],
                        state_names={'Starts': ['True', 'False'],
                                     'Ignition': ['True', 'False'],
                                     'Gas': ['True', 'False'],
                                     'StarterMotor': ['True', 'False'],
                                     'NotIcyWeather': ['True', 'False']})
cpd_moves = TabularCPD('Moves', 2, [[0.8, 0], [0.2, 1]], evidence=['Starts'], evidence_card=[2], state_names={'Moves': ['True', 'False'], 'Starts': ['True', 'False']})


car_model_updated.add_cpds(cpd_battery, cpd_radio, cpd_ignition, cpd_gas, cpd_moves, cpd_not_icy_weather, cpd_starter_motor, cpd_starts)

assert car_model_updated.check_model()

infer_updated = VariableElimination(car_model_updated)
radio_prob_given_car_not_start = infer_updated.query(variables=['Radio'], evidence={'Starts': 'False'})

print("Probability of Radio :", radio_prob_given_car_not_start)
