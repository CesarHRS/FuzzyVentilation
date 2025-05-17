import numpy as np
import matplotlib.pyplot as plt


# =============================================
# 1. Definição das funções de pertinência
# =============================================

def triangular(x, params):
    """Função triangular de pertinência"""
    a, b, c = params
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= a or x[i] >= c:
            y[i] = 0
        elif a < x[i] <= b:
            y[i] = (x[i] - a) / (b - a)
        elif b < x[i] < c:
            y[i] = (c - x[i]) / (c - b)
    return y


def gaussian(x, params):
    """Função gaussiana de pertinência"""
    mean, sigma = params
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def sigmoidal(x, params):
    """Função sigmoidal de pertinência"""
    a, c = params
    return 1 / (1 + np.exp(-a * (x - c)))


# =============================================
# 2. Universos de discurso e conjuntos fuzzy
# =============================================

# Entrada
temperature = np.arange(0, 51, 1)  # Temperatura (0-50°C)
humidity = np.arange(0, 101, 1)  # Umidade (0-100%)
people = np.arange(0, 11, 1)  # Pessoas (0-10)

# Saída
ventilation = np.arange(0, 101, 1)  # Ventilação (0-100%)

# Temperatura
low_temp = triangular(temperature, [0, 15, 25])
mid_temp = triangular(temperature, [20, 25, 30])
high_temp = triangular(temperature, [25, 35, 50])

# Humidade
low_hum = triangular(humidity, [0, 20, 50])
mid_hum = triangular(humidity, [30, 50, 70])
high_hum = triangular(humidity, [50, 80, 100])

# Pessoas
few_people = triangular(people, [0, 0, 4])
some_people = triangular(people, [2, 5, 8])
many_people = triangular(people, [6, 10, 10])

# Ventilação
low_vent = triangular(ventilation, [0, 0, 50])
mid_vent = triangular(ventilation, [20, 50, 80])
high_vent = triangular(ventilation, [50, 100, 100])

# =============================================
# 3. Regras fuzzy (Mamdani)
# =============================================

rules = [
    {'input': ['low', 'low', 'few'], 'output': 'low', 'op': 'and'},
    {'input': ['mid', 'mid', 'some'], 'output': 'mid', 'op': 'and'},
    {'input': ['high', 'high', 'many'], 'output': 'high', 'op': 'or'},
    {'input': ['mid', 'high', None], 'output': 'mid', 'op': 'and'},
    {'input': ['high', 'mid', None], 'output': 'high', 'op': 'and'}
]


# =============================================
# 4. Funções para inferência fuzzy manual
# =============================================

def fuzzify(value, universe, sets):
    """Calcula os valores de pertinência do valor de entrada"""
    idx = np.where(universe == value)[0][0]
    return {name: sets[name][idx] for name in sets}


def evaluate_rule(rule, membership):
    """Avalia uma regra fuzzy e retornando a força de ativação"""
    operands = []
    for i, var in enumerate(['temp', 'hum', 'people']):
        if rule['input'][i]:
            operands.append(membership[var][rule['input'][i]])

    if rule['op'] == 'and':
        return min(operands)
    else:  # 'or'
        return max(operands)


def aggregate_outputs(activated_rules):
    """Agrega as saídas usando o max"""
    aggregated = np.zeros(len(ventilation))
    for rule in activated_rules:
        output_set = globals()[f"{rule['output']}_vent"]
        clipped = np.fmin(rule['strength'], output_set)
        aggregated = np.fmax(aggregated, clipped)
    return aggregated


def defuzzify(aggregated):
    """Calcula o centroide para defuzzificação"""
    sum_moment = np.sum(ventilation * aggregated)
    sum_area = np.sum(aggregated)
    return sum_moment / sum_area if sum_area != 0 else 0


# =============================================
# 5. Função de simulação
# =============================================

def simulate(temp, hum, ppl):
    # Fuzzificação
    membership = {
        'temp': fuzzify(temp, temperature, {
            'low': low_temp,
            'mid': mid_temp,
            'high': high_temp
        }),
        'hum': fuzzify(hum, humidity, {
            'low': low_hum,
            'mid': mid_hum,
            'high': high_hum
        }),
        'people': fuzzify(ppl, people, {
            'few': few_people,
            'some': some_people,
            'many': many_people
        })
    }

    # Avaliação das regras
    activated_rules = []
    for rule in rules:
        strength = evaluate_rule(rule, membership)
        if strength > 0:
            activated_rules.append({
                'output': rule['output'],
                'strength': strength
            })

    # Agregação
    aggregated = aggregate_outputs(activated_rules)

    # Defuzzificação
    output = defuzzify(aggregated)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ventilation, low_vent, 'b--', linewidth=0.5, label='Fraca')
    plt.plot(ventilation, mid_vent, 'g--', linewidth=0.5, label='Moderada')
    plt.plot(ventilation, high_vent, 'r--', linewidth=0.5, label='Forte')
    plt.fill_between(ventilation, aggregated, alpha=0.2, color='cyan')
    plt.axvline(x=output, color='k', linestyle='--',
                label=f'Centroide: {output:.2f}%')
    plt.title(f'Cenário: {temp}°C, {hum}%, {ppl} pessoas\nIntensidade de Ventilação')
    plt.xlabel('Intensidade de Ventilação (%)')
    plt.ylabel('Grau de Pertinência')
    plt.legend()
    plt.grid(True)
    plt.show()

    return output


# =============================================
# 6. Teste
# =============================================

scenarios = [
    (18, 30, 2),  # Frio e seco com poucas pessoas
    (32, 75, 9),  # Quente e úmido com muitas pessoas
    (25, 85, 5),  # Temperatura média e umidade alta
    (28, 60, 3),  # Levemente quente com umidade média
    (35, 90, 7)   # Muito quente e úmido
]

print("=" * 50)
print("Sistema de Controle Fuzzy de Ventilação")
print("=" * 50)

for temp, hum, ppl in scenarios:
    result = simulate(temp, hum, ppl)
    print(f"\nCenário: {temp}°C | {hum}% | {ppl} pessoas")
    print(f"-> Intensidade de ventilação calculada: {result:.2f}%")
    print("-" * 50)