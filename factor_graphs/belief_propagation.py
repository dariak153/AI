import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def compute_probability(day, umbrella_data, rain_inference):
    evidence_dict = {}
    for i in range(day + 1):
        evidence_dict['Umbrella_' + str(i)] = umbrella_data[i]
    prob = rain_inference.query(['Rain_' + str(i)], evidence=evidence_dict)
    return prob

def main():
    edge_list = []

    for i in range(9):
        edge_list.append(('Rain_' + str(i), 'Umbrella_' + str(i)))
        if i < 9:
            edge_list.append(('Rain_' + str(i), 'Rain_' + str(i + 1)))

    edge_list.append(('Rain_9', 'Umbrella_9'))

    rain_network = BayesianNetwork(edge_list)

    initial_rain_cpd = TabularCPD('Rain_0', 2, [[0.6], [0.4]], state_names={'Rain_0': [True, False]})
    rain_network.add_cpds(initial_rain_cpd)

    initial_umbrella_cpd = TabularCPD('Umbrella_0', 2, [[0.9, 0.2], [0.1, 0.8]], evidence_card=[2],
                                      evidence=['Rain_0'], state_names={'Umbrella_0': [True, False],
                                                                       'Rain_0': [True, False]})
    rain_network.add_cpds(initial_umbrella_cpd)

    for i in range(9):
        rain_cpd = TabularCPD('Rain_' + str(i + 1), 2, [[0.7, 0.3], [0.3, 0.7]], evidence_card=[2],
                              evidence=['Rain_' + str(i)], state_names={'Rain_' + str(i + 1): [True, False],
                                                                       'Rain_' + str(i): [True, False]})
        rain_network.add_cpds(rain_cpd)

        umbrella_cpd = TabularCPD('Umbrella_' + str(i + 1), 2, [[0.9, 0.2], [0.1, 0.8]], evidence_card=[2],
                                  evidence=['Rain_' + str(i + 1)], state_names={'Umbrella_' + str(i + 1): [True, False],
                                                                               'Rain_' + str(i + 1): [True, False]})
        rain_network.add_cpds(umbrella_cpd)

    print('Model validation :', rain_network.check_model())

    rain_inference = VariableElimination(rain_network)

    umbrella_observations = [True, True, False, True, True]
    for i in range(len(umbrella_observations)):
        prob = compute_probability(i, umbrella_observations, rain_inference)
        print('Probability of Rain on Day ' + str(i) + ':\n', prob)

    evidence_set_1 = {'Umbrella_0': True, 'Umbrella_1': True,
                      'Umbrella_2': False, 'Umbrella_3': True,
                      'Umbrella_4': True}

    prob = rain_inference.query(['Rain_5'], evidence_set_1)
    print('Probability of Rain on Day 5:\n', prob)

    prob = rain_inference.query(['Rain_8'], evidence_set_1)
    print('Probability of Rain on Day 8:\n', prob)


if __name__ == '__main__':
    main()
