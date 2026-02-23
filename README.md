# Otimizador Genético de Personagens D&D

Aplicação desenvolvida em Python utilizando Streamlit para demonstrar a aplicação de Algoritmos Genéticos na otimização de personagens de RPG.

## Integrantes

- Cauan Teixeira Machado
- João Felipe Quentino

## Descrição do Projeto

O sistema utiliza dados de personagens e monstros para simular batalhas e calcular o dano médio por rodada (DPR). 
O algoritmo genético busca maximizar esse valor ajustando atributos, classe, raça e arma.

## Tecnologias Utilizadas

- Python 3
- Streamlit
- Pandas
- NumPy
- Algoritmos Genéticos

## Como Executar

1. Instale as dependências:

```
pip install streamlit pandas numpy
```

2. Estruture os arquivos:

```
data/chars.csv
data/monsters.csv
```

3. Execute:

```
streamlit run app.py
```

## Estrutura do Projeto

- app.py → Aplicação principal
- data/ → Arquivos CSV utilizados
- Relatorio_Tecnico_Algoritmo_Genetico_DnD.pdf → Relatório técnico do projeto

## Observações

A função de aptidão prioriza o atributo correto de acordo com a arma escolhida, 
além de aplicar ajustes específicos para determinadas classes e raças.
