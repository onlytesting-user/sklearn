# Inteligência Artificial de Classificação

<p>Um simples projeto de uma IA de classificação baseada em aprendizado supervisionado. Seus dados são obtidos de um arquivo <i>.csv</i> com o auxílio da biblioteca <b>pandas</b> e seu treinamento é feito pela biblioteca <b>scikit-learn</b>, através da qual treinamos dois modelos diferentes — KNeighbors e RandomForest — para decidir qual vamos usar através da comparação de precisão entre ambos. A biblioteca <b>numpy</b> atua como um auxiliar para o teste de um dos modelos ser possível</p>

## Bibliotecas Usadas:

* #### Pandas 2.2.1:
    - Usada para ler o arquivo .csv de onde os dados de treino e teste foram obtidos. Além disso também é usado para fazer a divisão e exclusão de colunas da tabela para fins de otimização do aprendizado da IA.
* #### Sciki-Learn 1.4.1.post1:
    - o módulo <i><b>preprocessing</b></i> é usado para podermos usar o método <i>``LabelEncoder()``</i> e converter as colunas Education e Occupation do tipo <i>"object"</i> úteis para previsão para tipo <i>"int32"</i>. Dessa forma, possibilitamos o uso dessas colunas para o aprendizado da IA;

    - o módulo <i><b>model_selection</b></i> é usado para fazer a divisão entre dados de treino e teste usando o módulo <i>``train_test_split()``</i>. Definimos o tamanho do teste como 30% do total da base de dados utilizada, o que automaticamente definiu o tamanho da base de treino como 70% do total;

    - os módulos <i><b>neighbors</b></i> e <i><b>ensemble</b></i> são usados para podermos usar, respectivamente, os módulos <i>``KNeighborsClassifier()``</i> e <i>``RandomForestClassifier()``</i> para treinarmos dois modelos diferentes de IA para previsão de forma simultânea;

    - o módulo <i><b>metrics</b></i> é usado para podermos medir a acurácia dos modelos utilizados através do método <i>``accuracy_score()``</i> e, assim, podermos definir qual o melhor modelo a ser usado como definitivo para previsões futuras.

* #### NumPy 1.26.4:
    - usada apenas para converter para números Python os dados de teste da biblioteca KNeighbors através do método <i>``to_numpy()``</i>.

## Código

- <b>Células 1-3:</b> Importação do Pandas, importação e visualização dos dados do arquivo <i>.csv</i> com os dados e filtragem de colunas da tabela;

- <b>Células 4-5:</b> Importação do módulo preprocessing da biblioteca scikit-learn e codificação dos dados, passando-os de <i>"object"</i> para <i>"int32"</i>;

- <b>Células 6-7:</b> Importação dos módulos neighbors e ensemble e de seus respectivos módulos: <i>KNeighbors</i> e <i>RandomForest</i>. Treino e teste da ambos os modelos de IA;

- <b>Células 8-10:</b> Usando a IA anterior com outra base de dados utilizando o modelo com maior precisão, que no caso foi o <i>KNeighbors</i>. A IA consegue prever os dados dessa segunda base de dados com 100% de acurácia.

##### <p>1-3:</p>

```py
import pandas as pd

df = pd.read_csv("ContosoCustomer.csv", sep=";", decimal=",")

useless_columns = ["CustomerName", "Age", "MaritalStatus", "Gender", "CustomerType"]

df = df.drop(columns=useless_columns)
```

##### <p>4-5:</p>

```py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

coder = LabelEncoder()

df["Education"] = coder.fit_transform(df["Education"])
df["Occupation"] = coder.fit_transform(df["Occupation"])

y = df["CustomerScore"]
x = df.drop(columns="CustomerScore")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
```

##### <p>6-7:</p>

```py
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

k_neighbors_model = KNeighborsClassifier()
random_forest_model = RandomForestClassifier()

k_neighbors_model.fit(x_train, y_train)
random_forest_model.fit(x_train, y_train)

k_neighbors_predicts = k_neighbors_model.predict(x_test)
random_forest_predicts = random_forest_model.predict(x_test.to_numpy())

from sklearn.metrics import accuracy_score

k_neighbors_accuracy = accuracy_score(y_test, k_neighbors_predicts, normalize=True)
random_forest_accuracy = accuracy_score(y_test, random_forest_predicts, normalize=True)

print(f"{k_neighbors_accuracy:.2%}")
print(f"{random_forest_accuracy:.2%}")
```

##### <p>8-10:</p>

```py
new_customers = pd.read_csv("ContosoNewCustomer.csv", sep=";", decimal=",")

new_customers = new_customers.drop(columns=useless_columns)

new_customers["Education"] = coder.fit_transform(new_customers["Education"])
new_customers["Occupation"] = coder.fit_transform(new_customers["Occupation"])

predicts = random_forest_model.predict(new_customers)
print(predicts)
```

## Documentação:

Biblioteca | Guia
---|---
Pandas | [pandas_documentation](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)
Scikit-Learn | [sklearn_documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
NumPy | [numpy_documentation](https://numpy.org/doc/1.26/user/index.html#user)
