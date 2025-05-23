import numpy as np
import matplotlib.pyplot as plt

# =============================================
# 1. Definição das funções de pertinência
# =============================================

def triangular(x, params):
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
    mean, sigma = params
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def sigmoidal(x, params):
    a, c = params
    return 1 / (1 + np.exp(-a * (x - c)))

# =============================================
# 2. Universos de discurso
# =============================================

temperature = np.arange(0, 51, 1)
humidity = np.arange(0, 101, 1)
people = np.arange(0, 11, 1)
ventilation = np.arange(0, 101, 1)

# =============================================
# 3. Parâmetros dos conjuntos fuzzy para cada tipo
# =============================================

# Temperatura
tri_temp = [
    [0, 15, 25],   # low
    [20, 25, 30],  # mid
    [25, 35, 50]   # high
]
gau_temp = [
    [10, 6],       # low (mean, sigma)
    [25, 3],       # mid
    [40, 8]        # high
]
sig_temp = [
    [-0.4, 15],    # low (a, c)
    [0.4, 30],     # high
    [2, 25],       # mid (usando sigmoidal para simular o "meio")
]

# Umidade
tri_hum = [
    [0, 20, 50],   # low
    [30, 50, 70],  # mid
    [50, 80, 100]  # high
]
gau_hum = [
    [15, 10],      # low
    [50, 10],      # mid
    [85, 10]       # high
]
sig_hum = [
    [-0.2, 20],    # low
    [0.2, 80],     # high
    [2, 50],       # mid
]

# Pessoas
tri_pess = [
    [0, 0, 4],     # few
    [2, 5, 8],     # some
    [6, 10, 10]    # many
]
gau_pess = [
    [1, 2],        # few
    [5, 2],        # some
    [9, 1.5]       # many
]
sig_pess = [
    [-2, 2],       # few
    [2, 8],        # many
    [2, 5],        # some
]

# Ventilação
tri_vent = [
    [0, 0, 50],    # low
    [20, 50, 80],  # mid
    [50, 100, 100] # high
]
gau_vent = [
    [10, 15],      # low
    [50, 15],      # mid
    [90, 10]       # high
]
sig_vent = [
    [-0.12, 20],   # low
    [0.12, 80],    # high
    [0.2, 50],     # mid
]

# =============================================
# 4. Regras fuzzy (Mamdani)
# =============================================

rules = [
    {'input': ['low', 'low', 'few'], 'output': 'low', 'op': 'and'},
    {'input': ['mid', 'mid', 'some'], 'output': 'mid', 'op': 'and'},
    {'input': ['high', 'high', 'many'], 'output': 'high', 'op': 'or'},
    {'input': ['mid', 'high', None], 'output': 'mid', 'op': 'and'},
    {'input': ['high', 'mid', None], 'output': 'high', 'op': 'and'}
]

# =============================================
# 5. Funções para inferência fuzzy manual
# =============================================

def fuzzify(value, universe, sets):
    idx = np.where(universe == value)[0][0]
    return {name: sets[name][idx] for name in sets}

def evaluate_rule(rule, membership):
    operands = []
    for i, var in enumerate(['temp', 'hum', 'people']):
        if rule['input'][i]:
            operands.append(membership[var][rule['input'][i]])
    if rule['op'] == 'and':
        return min(operands)
    else:  # 'or'
        return max(operands)

def aggregate_outputs(activated_rules, output_sets):
    aggregated = np.zeros(len(ventilation))
    for rule in activated_rules:
        output_set = output_sets[rule['output']]
        clipped = np.fmin(rule['strength'], output_set)
        aggregated = np.fmax(aggregated, clipped)
    return aggregated

def defuzzify(aggregated):
    sum_moment = np.sum(ventilation * aggregated)
    sum_area = np.sum(aggregated)
    return sum_moment / sum_area if sum_area != 0 else 0

# =============================================
# 6. Função para gerar os conjuntos de pertinência
# =============================================

def build_sets(tipo):
    if tipo == 'triangular':
        temp_sets = {
            'low': triangular(temperature, tri_temp[0]),
            'mid': triangular(temperature, tri_temp[1]),
            'high': triangular(temperature, tri_temp[2])
        }
        hum_sets = {
            'low': triangular(humidity, tri_hum[0]),
            'mid': triangular(humidity, tri_hum[1]),
            'high': triangular(humidity, tri_hum[2])
        }
        pess_sets = {
            'few': triangular(people, tri_pess[0]),
            'some': triangular(people, tri_pess[1]),
            'many': triangular(people, tri_pess[2])
        }
        vent_sets = {
            'low': triangular(ventilation, tri_vent[0]),
            'mid': triangular(ventilation, tri_vent[1]),
            'high': triangular(ventilation, tri_vent[2])
        }
    elif tipo == 'gaussian':
        temp_sets = {
            'low': gaussian(temperature, gau_temp[0]),
            'mid': gaussian(temperature, gau_temp[1]),
            'high': gaussian(temperature, gau_temp[2])
        }
        hum_sets = {
            'low': gaussian(humidity, gau_hum[0]),
            'mid': gaussian(humidity, gau_hum[1]),
            'high': gaussian(humidity, gau_hum[2])
        }
        pess_sets = {
            'few': gaussian(people, gau_pess[0]),
            'some': gaussian(people, gau_pess[1]),
            'many': gaussian(people, gau_pess[2])
        }
        vent_sets = {
            'low': gaussian(ventilation, gau_vent[0]),
            'mid': gaussian(ventilation, gau_vent[1]),
            'high': gaussian(ventilation, gau_vent[2])
        }
    elif tipo == 'sigmoidal':
        temp_sets = {
            'low': sigmoidal(temperature, sig_temp[0]),
            'mid': sigmoidal(temperature, sig_temp[2]),
            'high': sigmoidal(temperature, sig_temp[1])
        }
        hum_sets = {
            'low': sigmoidal(humidity, sig_hum[0]),
            'mid': sigmoidal(humidity, sig_hum[2]),
            'high': sigmoidal(humidity, sig_hum[1])
        }
        pess_sets = {
            'few': sigmoidal(people, sig_pess[0]),
            'some': sigmoidal(people, sig_pess[2]),
            'many': sigmoidal(people, sig_pess[1])
        }
        vent_sets = {
            'low': sigmoidal(ventilation, sig_vent[0]),
            'mid': sigmoidal(ventilation, sig_vent[2]),
            'high': sigmoidal(ventilation, sig_vent[1])
        }
    return temp_sets, hum_sets, pess_sets, vent_sets

# =============================================
# 7. Plot
# =============================================

def simulate_and_plot(temp, hum, ppl, tipo):
    temp_sets, hum_sets, pess_sets, vent_sets = build_sets(tipo)
    membership = {
        'temp': fuzzify(temp, temperature, temp_sets),
        'hum': fuzzify(hum, humidity, hum_sets),
        'people': fuzzify(ppl, people, pess_sets)
    }
    # Avaliação das regras
    activated_rules = []
    for rule in rules:
        strength = evaluate_rule(rule, membership)
        if strength > 0:
            activated_rules.append({'output': rule['output'], 'strength': strength})
    # Agregação
    aggregated = aggregate_outputs(activated_rules, vent_sets)
    # Defuzzificação
    output = defuzzify(aggregated)
    # Plot
    plt.plot(ventilation, vent_sets['low'], 'b--', linewidth=0.5, label='Fraca' if tipo == 'triangular' else None)
    plt.plot(ventilation, vent_sets['mid'], 'g--', linewidth=0.5, label='Moderada' if tipo == 'triangular' else None)
    plt.plot(ventilation, vent_sets['high'], 'r--', linewidth=0.5, label='Forte' if tipo == 'triangular' else None)
    plt.fill_between(ventilation, aggregated, alpha=0.2, color='cyan')
    plt.axvline(x=output, color='k', linestyle='--', label=f'Centroide: {output:.2f}%')
    return output

# =============================================
# 8. Teste 
# =============================================

scenarios = [
    (18, 30, 2),  # Frio e seco com poucas pessoas
    (32, 75, 9),  # Quente e úmido com muitas pessoas
    (25, 85, 5),  # Temperatura média e umidade alta
    (28, 60, 3),  # Levemente quente com umidade média
    (35, 90, 7)   # Muito quente e úmido
]

tipos = ['triangular', 'gaussian', 'sigmoidal']
nomes = {'triangular': 'Triangular', 'gaussian': 'Gaussiana', 'sigmoidal': 'Sigmoidal'}

print("=" * 50)
print("Sistema de Controle Fuzzy de Ventilação")
print("=" * 50)

for idx, (temp, hum, ppl) in enumerate(scenarios):
    plt.figure(figsize=(18, 4))
    saidas = []
    for i, tipo in enumerate(tipos):
        plt.subplot(1, 3, i+1)
        out = simulate_and_plot(temp, hum, ppl, tipo)
        saidas.append(out)
        plt.title(f"{nomes[tipo]}")
        plt.xlabel('Intensidade de Ventilação (%)')
        if i == 0:
            plt.ylabel('Grau de Pertinência')
        plt.grid(True)
        plt.legend()
    plt.suptitle(f'Cenário {idx+1}: {temp}°C, {hum}%, {ppl} pessoas\nComparação das Funções de Pertinência')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
    print(f"\nCenário: {temp}°C | {hum}% | {ppl} pessoas")
    print(f"-> Triangular: {saidas[0]:.2f}% | Gaussiana: {saidas[1]:.2f}% | Sigmoidal: {saidas[2]:.2f}%")
    print("-" * 50)
