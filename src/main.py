import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Definição das variáveis fuzzy
temperature = ctrl.Antecedent(np.arange(15, 41, 1), 'temperatura')
humidity = ctrl.Antecedent(np.arange(20, 101, 1), 'umidade')
people = ctrl.Antecedent(np.arange(0, 11, 1), 'pessoas')
ventilation = ctrl.Consequent(np.arange(0, 101, 1), 'ventilacao')

# 2. Funções de pertinência
temperature['baixa'] = fuzz.trimf(temperature.universe, [15, 15, 25])
temperature['media'] = fuzz.trimf(temperature.universe, [20, 25, 30])
temperature['alta'] = fuzz.trimf(temperature.universe, [25, 35, 40])

humidity['baixa'] = fuzz.trimf(humidity.universe, [20, 20, 50])
humidity['media'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['alta'] = fuzz.trimf(humidity.universe, [50, 80, 100])

people['poucas'] = fuzz.trimf(people.universe, [0, 0, 4])
people['moderadas'] = fuzz.trimf(people.universe, [2, 5, 8])
people['muitas'] = fuzz.trimf(people.universe, [6, 10, 10])

ventilation['fraca'] = fuzz.trimf(ventilation.universe, [0, 0, 50])
ventilation['moderada'] = fuzz.trimf(ventilation.universe, [20, 50, 80])
ventilation['forte'] = fuzz.trimf(ventilation.universe, [50, 100, 100])

# 3. Visualização das funções de pertinência
"""
temperature.view()
humidity.view()
people.view()
ventilation.view()
plt.show()
"""

# 4. Definição das regras fuzzy
rule1 = ctrl.Rule(temperature['baixa'] & humidity['baixa'] & people['poucas'], ventilation['fraca'])
rule2 = ctrl.Rule(temperature['media'] & humidity['media'] & people['moderadas'], ventilation['moderada'])
rule3 = ctrl.Rule(temperature['alta'] | humidity['alta'] | people['muitas'], ventilation['forte'])
rule4 = ctrl.Rule(temperature['media'] & humidity['alta'], ventilation['moderada'])
rule5 = ctrl.Rule(temperature['alta'] & humidity['media'], ventilation['forte'])

# 5. Criação do sistema de controle
system = ctrl.ControlSystemSimulation(ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5]))


# 6. Simulação com os cenários definidos previamente
def simulate(temperature, humidity, people):
    system.input['temperatura'] = temperature
    system.input['umidade'] = humidity
    system.input['pessoas'] = people

    try:
        system.compute()
        print(f"\nCenário - Temperatura: {temperature}°C, Umidade: {humidity}%, Pessoas: {people}")
        print("Intensidade de ventilação:", system.output['ventilacao'])

        # Visualização do resultado
        ventilation.view(sim=system)
        plt.show()

    except Exception as e:
        print(f"Erro ao processar cenário: {e}")


# Teste
cenarios = [
    (18, 30, 2),  # Frio e seco com poucas pessoas
    (32, 75, 9),  # Quente e úmido com muitas pessoas
    (25, 85, 5),  # Temperatura média e umidade alta
    (28, 60, 3),  # Levemente quente com umidade média
    (35, 90, 7)   # Muito quente e úmido
]

for cenario in cenarios:
    simulate(*cenario)