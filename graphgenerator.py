import networkx as nx
import pandas as pd
import random
import string

def gerar_nomes_nos(n):
    """ Gera rótulos de nós como A, B, C, ... """
    alfabeto = list(string.ascii_uppercase)
    return alfabeto[:n]

def gerar_grafo_euleriano(nos, arestas):
    """ Gera um grafo Euleriano aleatório """
    nomes_nos = gerar_nomes_nos(nos)
    G = nx.Graph()

    # Adiciona arestas de forma que o grafo seja conexo
    for _ in range(arestas):
        u, v = random.sample(nomes_nos, 2)
        peso = random.randint(1, 10)
        G.add_edge(u, v, weight=peso)

    # Garante que todos os nós tenham grau par (para ser Euleriano)
    graus_impares = [n for n in G.nodes if G.degree[n] % 2 != 0]
    while graus_impares:
        u = graus_impares.pop()
        v = graus_impares.pop()
        peso = random.randint(1, 10)
        G.add_edge(u, v, weight=peso)

    return G

def salvar_grafo_csv(G, nome_arquivo):
    """ Salva o grafo no formato CSV com colunas (u, v, w) """
    df = pd.DataFrame([(u, v, G[u][v]['weight']) for u, v in G.edges], columns=["u", "v", "w"])
    df.to_csv(nome_arquivo, index=False)
    print(f"Grafo salvo em {nome_arquivo}")


# Criar e salvar dois grafos Eulerianos
G1 = gerar_grafo_euleriano(nos=5, arestas=8)
G2 = gerar_grafo_euleriano(nos=6, arestas=10)

salvar_grafo_csv(G1, "grafo1.csv")
salvar_grafo_csv(G2, "grafo2.csv")

