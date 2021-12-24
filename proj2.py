# Alexandre Souza Costa Oliveira
# 170098168
# Universidade de Brasilia
# Bibliotecas de terceiros: sklearn, numpy, pandas, matplotlib, seaborn
# Feito em ambiente Windows 10, Python 3.9.7
# Observacao: ao executar, pode demorar um pouco para rodar, por conta das bibliotecas

import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Funcao que plota 4 retas para cada n_estimator
# Nao utlizando essa funcao ainda
def PlotarTodos(acuracia, precisao, revocacao, f1):
    grafico = pd.DataFrame({
        "acuracia": acuracia,
        "precisao": precisao,
        "revocacao": revocacao,
        "f1": f1
    })

    grafico.plot()
    plt.show()

# Funcao que plota um grafico com os resultados
def PlotarGraficoUnico(acuracia, precisao, revocacao, f1):
    grafico_resultados = pd.DataFrame({
        "resultado": list(["acuracia", "precisao", "revocacao", "med. f1"]),
        "valor": list([acuracia, precisao, revocacao, f1])
    })

    sns.barplot(x = grafico_resultados.resultado, y = grafico_resultados.valor)
    plt.xlabel("")
    plt.ylabel("")
    plt.title("Resultados da Floresta")
    plt.xticks(rotation=0, horizontalalignment="center", fontweight="light", fontsize="xx-small")
    plt.show()

def main():

    # le arquivo cvs
    data = pd.read_csv("diabetes.csv")
    
    # X fica apenas com as features
    X = data.drop(["Outcome"], axis=1)

    # Y fica apenas com as classes de cada linha
    y = data["Outcome"]

    # Normaliza os dados
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Realiza a divisao dos dados com 20% para teste e 80% para treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80)

    estimador = int(input("Quantos estimadores ? ( 0 para usar SQRT )"))

    if(estimador == 0):
        estimador = int(math.sqrt(768))
   
    # Instancia uma floresta randomica de classificacao
    floresta = RandomForestClassifier(n_estimators=estimador)

    # Treina a floresta / cria as arvores
    floresta.fit(X_train, y_train)

    # Cria a predicao para os dados de testes
    y_pred = floresta.predict(X_test)

    # Calcula os resultados de acuracia, precisao, revocacao, medida f1 e matriz de confusao
    # funcoes obtidas atraves da biblioteca sklearn.metrics
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    revocacao = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusao = confusion_matrix(y_test, y_pred)

    print("\nResultados da Floresta Aleatoria: \n")

    # Acuracia dos testes
    print("Acuracia: ", acuracia)
    print("Precisao: ", precisao)
    print("Revocacao: ", revocacao)
    print("Medida F1: ", f1)
    print("Confusao: ", confusao)

    # Plota o Grafico de Resultados
    PlotarGraficoUnico(acuracia, precisao, revocacao, f1)

    # Plota o Grafico de Confusao
    plot_confusion_matrix(floresta, X_test, y_test)
    plt.show()

    # Cria uma tabela com as features mais importantes
    # Com base nisso, podemos remover os menos importantes para tentar chegar em resultados mais precisos
    importantes = pd.DataFrame({
        "feature": list(X.columns),
        "importance": floresta.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeatures importantes: \n")
    print(importantes)

    # Plota e mostra um grafico com as features mais importantes
    sns.barplot(x=importantes.feature, y=importantes.importance)

    plt.xlabel("Features")
    plt.ylabel("Importância")
    plt.title("Gráfico de Features mais importantes")
    plt.xticks(rotation=0, horizontalalignment="center", fontweight="light", fontsize="xx-small")
    plt.show()

main()