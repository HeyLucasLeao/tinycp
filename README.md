Coletando informações do workspace

# Documentação do Projeto CP-Study

## Índice

1. Introdução
2. Instalação
3. Estrutura do Projeto
4. Utilização
5. Referências

## Introdução

O projeto **CP-Study** é uma ferramenta para análise e visualização de classificadores conformais binários utilizando a metodologia Out-of-Bag (OOB) com um classificador de floresta aleatória como modelo subjacente. Ele inclui funcionalidades para gerar curvas de eficiência, confiabilidade, histogramas de pontuações previstas, matrizes de confusão e gráficos de densidade de probabilidade Beta.

## Instalação

Para instalar o projeto, siga os passos abaixo:

1. Clone o repositório:
    ```sh
    git clone <URL_DO_REPOSITORIO>
    cd cp-study
    ```

2. Instale as dependências utilizando o Poetry:
    ```sh
    poetry install
    ```

## Estrutura do Projeto

A estrutura do projeto é a seguinte:

```
cp-study/
│
├── pyproject.toml
├── tinycp/
│   ├── binary/
│   │   ├── __init__.py
│   │   ├── cp.py
│   │   └── mcp.py
│   └── utils/
│       └── plotly_utils.py
```

### Arquivos Principais

- **pyproject.toml**: Arquivo de configuração do Poetry, contendo as dependências do projeto.
- **tinycp/binary/cp.py**: Implementação do classificador conformal binário utilizando a metodologia OOB.
- **tinycp/binary/mcp.py**: Implementação do classificador conformal binário modificado utilizando a metodologia OOB.
- **tinycp/utils/plotly_utils.py**: Funções utilitárias para visualização de dados utilizando Plotly.

## Utilização

### Classificador Conformal Binário

Para utilizar o classificador conformal binário, siga os passos abaixo:

1. Importe a classe 

WrapperOOBBinaryConformalClassifier

:
    ```python
    from tinycp.binary.cp import WrapperOOBBinaryConformalClassifier
    ```

2. Treine o classificador com um modelo de floresta aleatória:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Gerar dados de exemplo
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Treinar o modelo de floresta aleatória
    rf = RandomForestClassifier(oob_score=True, random_state=42)
    rf.fit(X, y)

    # Criar o classificador conformal
    clf = WrapperOOBBinaryConformalClassifier(rf)
    clf.fit(y)
    ```

3. Faça previsões utilizando o classificador conformal:
    ```python
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    ```

### Visualização de Dados

Para gerar visualizações utilizando as funções utilitárias do Plotly, siga os passos abaixo:

1. Importe as funções do módulo `plotly_utils`:
    ```python
    from tinycp.utils.plotly_utils import (
        efficiency_curve,
        reliability_curve,
        histogram,
        confusion_matrix,
        beta_pdf_with_cdf_fill
    )
    ```

2. Gere as visualizações desejadas:
    ```python
    # Curva de eficiência
    fig = efficiency_curve(clf, X)
    
    # Curva de confiabilidade
    fig = reliability_curve(clf, X, y)
    
    # Histograma de pontuações previstas
    fig = histogram(clf, X)
    
    # Matriz de confusão
    fig = confusion_matrix(clf, X, y)
    
    # Gráfico de densidade de probabilidade Beta
    fig = beta_pdf_with_cdf_fill(alpha=2, beta_param=5)
    ```

## Referências

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Plotly](https://plotly.com/python/)
- [Poetry](https://python-poetry.org/)
- [Venn Abers](https://github.com/donlnz/venn-abers)

Para mais informações, consulte a documentação oficial das bibliotecas utilizadas.