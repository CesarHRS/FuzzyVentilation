# **Sistema de Controle Fuzzy para Ventilação de Ambientes** 

**Autores**: César Henrique Resende Soares e Mateus Henrique Pereira  
**Email**: cesar@cefetmg.aluno.com e mateushenriqpereira02@gmail.com  
**Disciplina**: Inteligência Computacional  
**Professor**: Alisson Marques da Silva  
**Instituição**: CEFET-MG  

---

## **1. Objetivo**

Implementar um **sistema de controle fuzzy tipo Mamdani** para regular a intensidade de ventilação em ambientes fechados, considerando:

- Conforto térmico
- Eficiência energética
- Número de ocupantes

---

## **2. Funcionamento**

### **2.1. Entradas do Sistema**

| Variável          | Faixa de Valores | Conjuntos Fuzzy                 | 
|-------------------|------------------|---------------------------------|
| Temperatura (°C)  | 0 a 50           | `baixa`, `média`, `alta`        |
| Umidade (%)       | 0 a 100          | `baixa`, `média`, `alta`        |
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

1. **Fuzzificação**: Conversão das entradas em graus de pertinência usando funções triangulares definidas manualmente.  
2. **Avaliação de Regras**: Aplicação de operadores fuzzy (`min` para AND, `max` para OR).  
3. **Agregação**: Combinação das saídas das regras usando `np.fmax`.  
4. **Defuzzificação**: Cálculo do valor numérico final com método do centroide (média ponderada).  

### **3.2. Bibliotecas Utilizadas**

- `numpy`: Cálculos numéricos e manipulação de arrays.  
- `matplotlib`: Visualização gráfica dos conjuntos fuzzy e resultados.  

---

## **4. Como Executar**

### **4.1. Pré-requisitos**

Instale as bibliotecas necessárias:  
```bash
pip install numpy matplotlib
```

### **4.2. Execução do Sistema**

Entre na pasta src e execute o script Python diretamente:  
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

| Variável       | Conjunto Fuzzy | Pertinência | Cálculo                                               |
|----------------|----------------|-------------|-------------------------------------------------------|
| Temperatura    | baixa          | 0.3         | `triangular(18, [0,15,25]) = (18-15)/(25-15) = 0.3`   |
| Umidade        | baixa          | 0.666       | `triangular(30, [0,20,50]) = (50-30)/(50-20) ≈ 0.666` |
| Pessoas        | poucas         | 0.5         | `triangular(2, [0,0,4]) = (4-2)/(4-0) = 0.5`          |

#### **Passo 2: Regras Ativadas**

**Regra 1**:  
```python
IF Temperatura=baixa AND Umidade=baixa AND Pessoas=poucas THEN Ventilação=fraca  
Força de Ativação = min(0.3, 0.666, 0.5) = 0.3  
```

#### **Passo 3: Agregação**

A saída agregada é a interseção máxima entre as regras ativadas:  
```python
aggregated = np.fmax(np.fmin(0.3, low_vent), ...)  # Apenas Regra 1 ativa
```

#### **Passo 4: Defuzzificação**

Cálculo do centroide da área agregada:  
```python
centroid = ∑(ventilation * aggregated) / ∑(aggregated) ≈ 19.7%
```

## **Visualização**

![Saída Fuzzy](/img/example.png)  
*Gráfico gerado pelo código: Conjuntos fuzzy de saída (linhas tracejadas) e área agregada (azul).*

---

## **6. Resultados e Interpretação**

**Saída**: 19.7% → Ventilação **Fraca**  

**Interpretação**:  
- Ambiente frio (18°C) e seco (30%) com baixa ocupação.  
- Ventilação mínima mantém conforto térmico sem desperdício energético.  

---