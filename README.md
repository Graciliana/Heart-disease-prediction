# ❤️ Predição de Doenças Cardíacas com Machine Learning

## 1. Visão Geral

Este projeto foi desenvolvido como parte do **Projeto de Recuperação — Módulo 1 — Semana 14**, da disciplina de **Machine Learning e Visão Computacional**.

O objetivo é construir um **pipeline preditivo completo de Machine Learning** para auxiliar na identificação de pacientes com propensão a apresentar doenças cardíacas, utilizando informações clínicas e comportamentais disponíveis na base de dados.

O problema de negócio proposto consiste em prever a variável:

```text
HeartDisease
```

onde:

* `Yes` → paciente com indicação de doença cardíaca;
* `No` → paciente sem indicação de doença cardíaca.

O projeto aborda todas as etapas de um fluxo de Ciência de Dados: análise exploratória, tratamento e preparação dos dados, engenharia de atributos, balanceamento, escalonamento, modelagem, otimização de hiperparâmetros, avaliação e interpretação dos resultados.

> **Importante:** este projeto possui finalidade acadêmica e não deve ser utilizado como ferramenta de diagnóstico médico.

---

# 2. Problema de Negócio

Hospitais e planos de saúde podem utilizar dados históricos para identificar padrões associados à presença de doenças cardíacas.

O desafio consiste em desenvolver um modelo capaz de identificar pacientes com maior probabilidade de apresentar `HeartDisease = Yes`.

Entretanto, nesse contexto, **não basta obter uma alta acurácia**.

Os erros de classificação possuem impactos diferentes:

* **Falso Positivo (FP):** uma pessoa saudável é classificada como portadora de doença cardíaca;
* **Falso Negativo (FN):** uma pessoa com doença cardíaca é classificada como saudável.

O **Falso Negativo** apresenta especial relevância clínica, pois pode representar a não identificação de um paciente potencialmente em risco.

Por esse motivo, a escolha do modelo deve considerar não apenas a acurácia, mas também métricas como **Precision, Recall, F1-score e a matriz de confusão**.

O desafio proposto destaca justamente a necessidade de relacionar o desempenho estatístico do modelo aos impactos médicos, psicológicos, financeiros e operacionais envolvidos.

---

# 3. Objetivos

## 3.1 Objetivo Geral

Desenvolver e avaliar modelos de Machine Learning capazes de prever a presença de doença cardíaca a partir das características disponíveis na base de dados.

## 3.2 Objetivos Específicos

* realizar análise exploratória dos dados;
* identificar padrões e possíveis inconsistências;
* analisar a distribuição da variável alvo;
* identificar e tratar valores ausentes;
* identificar registros duplicados;
* analisar valores discrepantes;
* realizar engenharia de atributos;
* transformar variáveis categóricas em representação numérica;
* realizar separação estratificada entre treino e teste;
* aplicar balanceamento somente nos dados de treinamento;
* aplicar escalonamento quando necessário;
* desenvolver modelos KNN e Decision Tree;
* testar diferentes hiperparâmetros;
* diagnosticar possíveis situações de overfitting;
* utilizar XGBoost como modelo adicional de benchmark;
* comparar o desempenho dos modelos;
* analisar falsos positivos e falsos negativos;
* selecionar o modelo com melhor capacidade de generalização;
* elaborar uma recomendação orientada ao contexto de negócio.

---

# 4. Dataset

O projeto utiliza uma base de dados relacionada à saúde cardiovascular.

A variável alvo é:

```text
HeartDisease
```

### Variável alvo

| Variável       | Tipo       | Descrição                                                       |
| -------------- | ---------- | --------------------------------------------------------------- |
| `HeartDisease` | Categórica | Indica a presença (`Yes`) ou ausência (`No`) de doença cardíaca |

A base contém variáveis relacionadas a características demográficas, comportamentais, condições de saúde e indicadores de qualidade de vida.

---

# 5. Dicionário de Dados

| Variável           | Descrição                                  | Tipo       |
| ------------------ | ------------------------------------------ | ---------- |
| `HeartDisease`     | Indicador de doença cardíaca               | Categórica |
| `BMI`              | Índice de Massa Corporal                   | Numérica   |
| `Smoking`          | Histórico de tabagismo                     | Categórica |
| `AlcoholDrinking`  | Consumo de álcool                          | Categórica |
| `Stroke`           | Histórico de acidente vascular cerebral    | Categórica |
| `PhysicalHealth`   | Número de dias de saúde física debilitada  | Numérica   |
| `MentalHealth`     | Número de dias de saúde mental debilitada  | Numérica   |
| `DiffWalking`      | Dificuldade para caminhar ou subir escadas | Categórica |
| `Sex`              | Sexo informado                             | Categórica |
| `AgeCategory`      | Categoria de idade                         | Categórica |
| `Race`             | Raça/cor declarada                         | Categórica |
| `Diabetic`         | Indicador relacionado a diabetes           | Categórica |
| `PhysicalActivity` | Indicador de atividade física              | Categórica |
| `GenHealth`        | Avaliação geral da saúde                   | Categórica |
| `SleepTime`        | Horas de sono                              | Numérica   |
| `Asthma`           | Indicador de asma                          | Categórica |
| `KidneyDisease`    | Indicador de doença renal                  | Categórica |
| `SkinCancer`       | Indicador de câncer de pele                | Categórica |

### Feature Engineering

Foi criada uma nova variável:

```text
WeakenedHealthDays
```

calculada por:

```text
WeakenedHealthDays = PhysicalHealth + MentalHealth
```

Essa feature representa a soma dos dias de saúde física e mental debilitada no período analisado.

A criação dessa variável atende à etapa obrigatória de Feature Engineering definida no desafio.

---

# 6. Metodologia

O projeto foi estruturado em etapas independentes para facilitar a reprodutibilidade e a manutenção.

```text
Raw Data
   │
   ▼
EDA
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Train/Test Split
   │
   ▼
Encoding
   │
   ├───────────────┐
   ▼               ▼
KNN              Tree/XGBoost
   │               │
Scaling         No Scaling
   │               │
   └───────┬───────┘
           ▼
         SMOTE
           │
           ▼
       Modeling
           │
           ▼
       Evaluation
           │
           ▼
   Business Verdict
```

---

# 7. Análise Exploratória — EDA

A primeira etapa foi dedicada à compreensão da estrutura e da qualidade dos dados.

Foram realizadas análises de:

* dimensões da base;
* tipos de dados;
* estatísticas descritivas;
* valores ausentes;
* registros duplicados;
* distribuição da variável alvo;
* variáveis numéricas;
* variáveis categóricas;
* valores discrepantes;
* correlação entre variáveis;
* distribuição das características dos pacientes.

Também foram utilizados gráficos para auxiliar a interpretação dos dados, incluindo:

* histogramas;
* boxplots;
* gráficos de frequência;
* análise da distribuição da variável alvo;
* matriz de correlação.

O objetivo da EDA foi identificar características dos dados que poderiam influenciar as etapas posteriores do pipeline.

---

# 8. Tratamento e Preparação dos Dados

## 8.1 Valores Ausentes

Os valores ausentes foram identificados e analisados considerando as características das variáveis.

A técnica de imputação foi selecionada com base na distribuição dos dados e na presença de possíveis valores extremos.

A justificativa detalhada encontra-se no **Notebook 02 — Preprocessing**.

---

## 8.2 Registros Duplicados

Foram identificados registros duplicados e realizada a remoção das duplicidades para evitar redundância na base de treinamento.

---

## 8.3 Outliers

Os valores discrepantes foram analisados utilizando estatísticas e visualizações, principalmente boxplots.

Embora alguns registros tenham sido identificados como outliers estatísticos, os valores foram considerados **plausíveis dentro do contexto da base de saúde**.

Por esse motivo, a decisão adotada foi:

> **Manter os valores extremos considerados plausíveis, sem remoção ou clipping.**

Essa decisão foi tomada de forma consciente, considerando que a identificação estatística de um outlier não significa necessariamente que o registro seja um erro.

---

# 9. Encoding das Variáveis Categóricas

As variáveis categóricas foram transformadas em representação numérica utilizando **One-Hot Encoding**.

O encoder foi ajustado somente sobre os dados de treinamento:

```text
X_train
    ↓
fit_transform()
```

e posteriormente aplicado ao conjunto de teste:

```text
X_test
    ↓
transform()
```

Essa abordagem evita vazamento de informações entre treinamento e teste.

Foi utilizada a configuração:

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

---

# 10. Feature Engineering

Foi criada a variável:

```text
WeakenedHealthDays
```

por meio da seguinte regra:

```python
WeakenedHealthDays = (
    PhysicalHealth +
    MentalHealth
)
```

Essa transformação permite representar conjuntamente os dias de saúde física e mental debilitada.

Os valores ausentes das variáveis de origem foram tratados antes da criação da nova variável, conforme requerido pelo desafio.

---

# 11. Separação entre Treino e Teste

Os dados foram divididos utilizando:

```python
test_size=0.20
```

e:

```python
stratify=y
```

A divisão resultou em:

* **80% dos dados para treinamento**;
* **20% dos dados para teste**.

A estratificação foi utilizada para preservar a proporção das classes da variável alvo nos dois conjuntos.

O `random_state` utilizado foi:

```python
42
```

garantindo reprodutibilidade.

---

# 12. Balanceamento das Classes

A variável `HeartDisease` apresenta forte desbalanceamento, com predominância da classe `No`.

Para reduzir o impacto desse desbalanceamento, foi utilizado:

```text
SMOTE
```

O balanceamento foi aplicado **exclusivamente ao conjunto de treinamento**.

O conjunto de teste não foi submetido ao SMOTE, preservando sua distribuição original para representar melhor o comportamento esperado em dados não vistos.

Essa decisão segue a regra metodológica do desafio para evitar **data leakage**.

---

# 13. Escalonamento

O escalonamento foi aplicado especificamente ao modelo **KNN**, utilizando:

```text
StandardScaler
```

A decisão se deve ao fato de que o KNN utiliza distância entre observações e, portanto, variáveis em escalas diferentes podem influenciar desproporcionalmente o resultado.

Para a **Decision Tree**, o escalonamento não é necessário, pois o algoritmo realiza divisões baseadas nos valores das variáveis e não em distâncias.

Essa distinção faz parte dos critérios de avaliação do projeto.

O XGBoost também será utilizado sem necessidade de escalonamento.

---

# 14. Modelagem

## 14.1 K-Nearest Neighbors — KNN

O primeiro modelo desenvolvido foi o KNN.

Foram avaliadas diferentes configurações do parâmetro:

```text
n_neighbors
```

Foram utilizados, no mínimo:

```text
K = 3
K = 5
K = 7
K = 9
```

Os resultados foram comparados entre os conjuntos de treinamento e teste.

---

## 14.2 Decision Tree

O segundo modelo foi uma Árvore de Decisão.

Foram avaliadas diferentes profundidades:

```text
max_depth = 3
max_depth = 5
max_depth = 7
max_depth = None
```

A comparação entre treinamento e teste permitiu analisar o comportamento do modelo conforme sua complexidade aumentava.

---

## 14.3 XGBoost — Modelo Adicional

Como experimento adicional, foi incluído o **XGBoost**.

O objetivo não é substituir os modelos exigidos pelo desafio, mas utilizar um algoritmo de boosting como benchmark adicional.

Foram avaliadas diferentes combinações de:

* `n_estimators`;
* `max_depth`;
* `learning_rate`.

O XGBoost será analisado quanto ao desempenho e à capacidade de generalização.

---

# 15. Avaliação dos Modelos

Os modelos serão avaliados utilizando métricas de classificação, incluindo:

* Accuracy;
* Precision;
* Recall;
* F1-score;
* ROC-AUC, quando aplicável;
* matriz de confusão.

Além das métricas no conjunto de teste, os resultados de treinamento serão analisados para identificar possíveis sinais de **overfitting**.

O objetivo não é simplesmente selecionar o modelo com maior desempenho no treinamento, mas encontrar uma configuração que apresente boa capacidade de generalização.

---

# 16. Diagnóstico de Overfitting

Para cada configuração foram comparados os resultados de treinamento e teste.

Uma diferença elevada entre as métricas de treino e teste pode indicar que o modelo está se ajustando excessivamente aos dados utilizados durante o treinamento.

A análise será utilizada para identificar:

```text
Boa generalização
        ↓
Train ≈ Test
```

versus:

```text
Possível Overfitting
        ↓
Train >> Test
```

No KNN, será analisado o efeito da alteração de `n_neighbors`.

Na Decision Tree, será observado o comportamento da métrica conforme `max_depth` aumenta.

O objetivo é selecionar uma configuração que represente um equilíbrio entre desempenho e generalização, conforme solicitado pelo desafio.

---

# 17. Resultados

> **Esta seção será atualizada após a execução definitiva dos Notebooks 03 e 04.**

## Melhor KNN

```text
Configuração: [preencher]
Accuracy: [preencher]
Precision: [preencher]
Recall: [preencher]
F1-score: [preencher]
ROC-AUC: [preencher]
```

## Melhor Decision Tree

```text
Configuração: [preencher]
Accuracy: [preencher]
Precision: [preencher]
Recall: [preencher]
F1-score: [preencher]
ROC-AUC: [preencher]
```

## XGBoost

```text
Configuração: [preencher]
Accuracy: [preencher]
Precision: [preencher]
Recall: [preencher]
F1-score: [preencher]
ROC-AUC: [preencher]
```

---

# 18. Matriz de Confusão

A matriz de confusão será utilizada para analisar:

```text
True Positive  (TP)
True Negative  (TN)
False Positive (FP)
False Negative (FN)
```

A análise dos erros será especialmente importante para a decisão final.

Em um contexto de saúde, o **Falso Negativo** pode representar um risco significativo, pois corresponde a um paciente potencialmente doente sendo classificado como saudável.

Por outro lado, Falsos Positivos podem gerar:

* exames adicionais;
* consultas desnecessárias;
* ansiedade psicológica;
* aumento de custos operacionais;
* utilização adicional de recursos de saúde.

Portanto, a decisão final não será baseada exclusivamente na Accuracy.

---

# 19. Veredito de Negócio

> **Esta seção será preenchida após a avaliação final dos dois modelos obrigatórios.**

A escolha do modelo em produção deverá considerar:

1. desempenho no conjunto de teste;
2. capacidade de generalização;
3. Recall da classe `HeartDisease = Yes`;
4. quantidade de Falsos Negativos;
5. quantidade de Falsos Positivos;
6. equilíbrio entre desempenho estatístico e impacto operacional.

O modelo recomendado será aquele que apresentar o melhor equilíbrio entre desempenho preditivo e risco associado aos erros de classificação.

O desafio solicita explicitamente que o resultado da matriz de confusão seja conectado à realidade operacional e que seja justificada a escolha do modelo que seria colocado em produção.

---

# 20. Estrutura do Projeto

```text
heart-disease-ml/
│
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── one_hot_encoder.pkl
│   └── standard_scaler.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── visualization.py
│   ├── preprocessing.py
│   └── evaluation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 21. Organização dos Notebooks

### Notebook 01 — EDA

Responsável por:

* compreensão dos dados;
* estatísticas descritivas;
* análise da variável alvo;
* análise das variáveis numéricas;
* análise das variáveis categóricas;
* outliers;
* correlações;
* principais insights.

### Notebook 02 — Preprocessing

Responsável por:

* limpeza;
* tratamento de nulos;
* duplicatas;
* outliers;
* Feature Engineering;
* encoding;
* Train/Test Split;
* scaling;
* SMOTE;
* salvamento dos dados processados.

### Notebook 03 — Modeling

Responsável por:

* KNN;
* Decision Tree;
* XGBoost como modelo adicional;
* testes de hiperparâmetros;
* comparação treino/teste;
* diagnóstico de overfitting;
* seleção das melhores configurações.

### Notebook 04 — Evaluation

Responsável por:

* avaliação final;
* Classification Report;
* matrizes de confusão;
* análise de FP e FN;
* comparação final;
* recomendação de negócio.

---

# 22. Boas Práticas de Engenharia

O projeto foi estruturado buscando seguir práticas de desenvolvimento utilizadas em projetos reais de Ciência de Dados.

### Reprodutibilidade

Foi utilizado:

```python
RANDOM_STATE = 42
```

nas etapas que possuem aleatoriedade.

### Prevenção de Data Leakage

Transformadores como encoder e scaler são ajustados somente com os dados de treinamento.

O SMOTE é aplicado exclusivamente ao treinamento.

### Separação de responsabilidades

Os notebooks são divididos por finalidade:

```text
EDA
↓
Preprocessing
↓
Modeling
↓
Evaluation
```

Essa separação facilita a manutenção e a auditoria do projeto.

### Versionamento

O projeto utiliza Git e GitHub para controle de versão.

As etapas devem ser organizadas preferencialmente em branches específicas, por exemplo:

```text
phase/eda
phase/data-prep
phase/modeling
phase/evaluation
```

e utilizar commits semânticos, conforme recomendado no desafio.

Exemplos:

```text
feat: adiciona análise exploratória
feat: implementa tratamento de valores ausentes
feat: adiciona feature WeakenedHealthDays
feat: implementa encoding categórico
feat: adiciona modelo KNN
feat: adiciona decision tree
feat: adiciona benchmark com XGBoost
fix: corrige data leakage no preprocessing
```

---

# 23. Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* XGBoost
* Jupyter Notebook
* Google Colab
* Visual Studio Code
* Git
* GitHub

---

# 24. Como Executar

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
```

## 2. Acessar o projeto

```bash
cd heart-disease-ml
```

## 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

## 4. Executar os notebooks

A sequência recomendada é:

```text
01_EDA.ipynb
       ↓
02_Preprocessing.ipynb
       ↓
03_Modeling.ipynb
       ↓
04_Evaluation.ipynb
```

---

# 25. Conclusão

Este projeto busca demonstrar não apenas a aplicação de algoritmos de Machine Learning, mas a construção de um **pipeline completo, reprodutível e metodologicamente fundamentado**.

A principal preocupação é evitar que métricas aparentemente elevadas sejam obtidas por meio de vazamento de dados ou de modelos excessivamente ajustados ao conjunto de treinamento.

A decisão final será baseada na combinação entre desempenho estatístico, capacidade de generalização e impacto dos erros de classificação.

Em um problema de saúde, a escolha do modelo deve considerar especialmente o risco associado à não identificação de pacientes potencialmente doentes.

---

# 26. Apresentação

O projeto será acompanhado de uma apresentação técnica de até **7 minutos**, conforme orientação do desafio.

A apresentação abordará:

1. objetivo do modelo e impacto da correta identificação de doenças cardíacas;
2. principais insights da EDA;
3. tratamento de valores ausentes e outliers;
4. identificação e prevenção de overfitting;
5. análise da matriz de confusão e justificativa do modelo recomendado.

Esses pontos correspondem ao roteiro definido para a apresentação do projeto.

---

## Autor

**Graciliana Kascher**

Projeto acadêmico de Machine Learning e Visão Computacional.

---

## Status do Projeto

🚧 **Em desenvolvimento**

* [x] Estrutura do projeto
* [x] Análise exploratória
* [x] Tratamento dos dados
* [x] Feature Engineering
* [x] Train/Test Split
* [x] Encoding
* [x] Balanceamento
* [x] Escalonamento
* [x] Estrutura de modelagem
* [ ] Execução final dos modelos
* [ ] Avaliação final
* [ ] Matriz de confusão
* [ ] Veredito de negócio
* [ ] Atualização dos resultados no README
* [ ] Apresentação em vídeo
* [ ] Entrega final
