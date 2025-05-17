# **Sistema de Controle Fuzzy para Ventilação de Ambientes**

**Autores**: César Henrique Resende Soares e Mateus Henrique Pereira

**Email**: cesar@cefetmg.aluno.com e mateushenriqpereira02@gmail.com 

**Disciplina**: Inteligência Computacional  

**Professor**: Alisson Marques da Silva

**Instituição**: CEFET-MG

---

## **1. Objetivo**

Implementar um **sistema de controle fuzzy tipo Mamdani** para regular a intensidade de ventilação em ambientes
fechados, considerando:

- Conforto térmico
- Eficiência energética
- Número de ocupantes

---

## **2. Funcionamento**

### **2.1. Entradas do Sistema**

| Variável          | Faixa de Valores | Conjuntos Fuzzy                 | 
|-------------------|------------------|---------------------------------|
| Temperatura (°C)  | 15 a 40          | `baixa`, `média`, `alta`        |
| Umidade (%)       | 20 a 100         | `baixa`, `média`, `alta`        |
| Número de Pessoas | 0 a 10           | `poucas`, `moderadas`, `muitas` |

### **2.2. Saída do Sistema**

| Variável                      | Faixa   | Conjuntos Fuzzy              |
|-------------------------------|---------|------------------------------|
| Intensidade de Ventilação (%) | 0 a 100 | `fraca`, `moderada`, `forte` |

### **2.3. Regras Fuzzy (Mamdani)**

1. **SE** Temperatura = Baixa **E** Umidade = Baixa **E** Pessoas = Poucas  
   **ENTÃO** Ventilação = Fraca
2. **SE** Temperatura = Média **E** Umidade = Média **E** Pessoas = Moderadas  
   **ENTÃO** Ventilação = Moderada
3. **SE** Temperatura = Alta **OU** Umidade = Alta **OU** Pessoas = Muitas  
   **ENTÃO** Ventilação = Forte
4. **SE** Temperatura = Média **E** Umidade = Alta  
   **ENTÃO** Ventilação = Moderada
5. **SE** Temperatura = Alta **E** Umidade = Média  
   **ENTÃO** Ventilação = Forte

---

## **3. Método Utilizado**

### **3.1. Fluxo de Inferência Fuzzy**

1. **Fuzzificação**: Conversão das entradas em graus de pertinência usando funções triangulares (`trimf`).
2. **Avaliação de Regras**: Aplicação de operadores fuzzy (`min` para AND, `max` para OR).
3. **Agregação**: Combinação das saídas das regras usando `np.fmax`.
4. **Defuzzificação**: Cálculo do valor numérico final com método do centroide.

É válido ressaltar que o AND (min) e OR (max), são implementados ao utilizar respectivamente o & e | na definição das 
regras, e internamente, o `skfuzzy` utiliza o np.maximum para calcular a união dos conjuntos.


### **3.2. Bibliotecas Utilizadas**

- `skfuzzy`: Implementação da lógica fuzzy
- `numpy`: Cálculos numéricos
- `matplotlib`: Visualização dos resultados

---

## **4. Como Executar**

### **4.1. Pré-requisitos**

Para executar o programa, é necessário a instalação das bibliotecas utilizadas, isso pode ser feito da seguinte maneira:

```bash
pip install numpy scikit-fuzzy matplotlib
```

### **4.2. Execução**
Vá para a pasta src e execute o arquivo main.py
```bash
cd src && python3 main.py
```

---

## **5. Análise Detalhada**

### **Cenário Analisado**
**Entradas**:
- Temperatura: 18°C
- Umidade: 30%
- Pessoas: 2

#### **Passo 1: Fuzzificação**

| Variável       | Conjunto Fuzzy | Pertinência | Cálculo |
|----------------|----------------|-------------|---------|
| Temperatura    | baixa          | 0.3         | `trimf(18, [15,15,25]) = (18-15)/(25-15) = 0.3` |
| Umidade        | baixa          | 0.666       | `trimf(30, [20,20,50]) = (50-30)/(50-20) ≈ 0.666` |
| Pessoas        | poucas         | 0.5         | `trimf(2, [0,0,4]) = (4-2)/(4-0) = 0.5` |

#### **Passo 2: Regras Ativadas**

**Regra 1**:
```python
IF Temperatura=baixa AND Umidade=baixa AND Pessoas=poucas THEN Ventilação=fraca
```

### Passo 3: Agregação

Como apenas uma regra foi ativada, a saída agregada é igual à saída da Regra 1:

```python
aggregated = np.fmin(0.3, ventilation['fraca'].mf)
```

## Passo 4: Defuzzificação

Para a defuzzificação, foi utilizada o método **Centroide**

```python
ventilation_output = fuzz.defuzz(ventilation.universe, aggregated, 'centroid')
```
- Resultado: 19.44% 

## Visualização

![Saída Fuzzy](/img/example.png)

*Área de defuzzificação com centroide em 19.44%*


## Passo 5: Interpretação Qualitativa


**Saída**: 19.44% → Classificada como **"Fraca"**

Essa resposta, é coerente com o ambiente frio (18°C) e seco (30%), e uma ventilação mínima manteria o conforto térmico
sem desperdício energético, além disso, também há poucas pessoas no ambiente, desse modo, o resultado faz sentido.

---




