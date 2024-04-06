# Inteligência Artificial de Classificação

<p>Um simples projeto de uma IA de classificação baseada em aprendizado supervisionado. Seus dados são obtidos de um arquivo *.csv* com o auxílio da biblioteca **pandas** e seu treinamento é feito pela biblioteca **scikit-learn**, através da qual treinamos dois modelos diferentes — KNeighbors e RandomForest — para decidir qual vamos usar através da comparação de precisão entre ambos. A biblioteca **numpy** atua como um auxiliar para o teste de um dos modelos ser possível</p>

## Bibliotecas Usadas:

* #### Pandas 2.2.1:
    - Usada para ler o arquivo .csv de onde os dados de treino e teste foram obtidos. Além disso também é usado para fazer a divisão e exclusão de colunas da tabela para fins de otimização do aprendizado da IA.
* #### Sciki-Learn 1.4.1.post1:
    - o módulo *preprocessing* é usado para podermos usar o método *LabelEncoder* e converter as colunas "Education" e "Occupation" do tipo *object* úteis para previsão para tipo *int*. Dessa forma, possibilitamos o uso dessas colunas para o aprendizado da IA;
    <br>
    - o módulo *model_selection* é usado para fazer a divisão entre dados de treino e teste usando o módulo *train_test_split*. Definimos o tamanho do teste como 30% do total da base de dados utilizada, o que automaticamente definiu o tamanho da base de treino como 70% do total;
    <br>
    - os módulos *neighbors* e *ensemble* são usados para podermos usar, respectivamente, os módulos *KNeighborsClassifier* e *RandomForestClassifier* para treinarmos dois modelos diferentes de IA para previsão de forma simultânea;
    <br>
    - o módulo *metrics* é usado para podermos medir a acurácia dos modelos utilizados através do método *accuracy_score* e, assim, podermos definir qual o melhor modelo a ser usado como definitivo para previsões futuras.

* #### NumPy 1.26.4:
    - usada apenas para converter para números Python os dados de teste da biblioteca KNeighbors através do método *to_numpy*.

## Código

- **Células 1-3:** Importação do Pandas, importação e visualização dos dados do arquivo *.csv* com os dados e filtragem de colunas da tabela;
<br>
- **Células 4-5:** Importação do módulo preprocessing da biblioteca scikit-learn e codificação dos dados, passando-os de *object* para *int*;
<br>
- **Células 6-7:** Importação dos módulos neighbors e ensemble e de seus respectivos módulos: *KNeighbors* e *RandomForest*. Treino e teste da ambos os modelos de IA;
<br>
- **Células 8-10:** Usando a IA anterior com outra base de dados utilizando o modelo com maior precisão, que no caso foi o *KNeighbors*. A IA consegue prever os dados dessa segunda base de dados com 100% de acurácia.
<br>

##### <p>1-3:</p>

```js
import pandas as pd

df = pd.read_csv("ContosoCustomer.csv", sep=";", decimal=",")

useless_columns = ["CustomerName", "Age", "MaritalStatus", "Gender", "CustomerType"]

df = df.drop(columns=useless_columns)
```
<br>

##### <p>4-7:</p>

```js
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
<br>

##### <p>8-10:</p>

```js
new_customers = pd.read_csv("ContosoNewCustomer.csv", sep=";", decimal=",")

new_customers = new_customers.drop(columns=useless_columns)

new_customers["Education"] = coder.fit_transform(new_customers["Education"])
new_customers["Occupation"] = coder.fit_transform(new_customers["Occupation"])

predicts = random_forest_model.predict(new_customers)
print(predicts)
```
<br>

## Documentação:

Biblioteca | Guia
---|---
Pandas | [pandas_documentation](https://pandas.pydata.org/docs/user_guide/index.html#user-guide)
Scikit-Learn | [sklearn_documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
NumPy | [numpy_documentation](https://numpy.org/doc/1.26/user/index.html#user)
