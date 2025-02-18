import csv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import permutations
import heapq

from graphgenerator import gerar_grafo_euleriano, salvar_grafo_csv


def read_graph_from_csv(file_path):
    G = nx.Graph()
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Pula o cabeçalho
        for row in reader:
            u, v, w = row
            G.add_edge(u, v, weight=int(w))
    return G


def plot_graph(G, title="Grafo"):
    pos = nx.spring_layout(G)  # Layout do grafo
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()


def dijkstra_path(G, source, target, weight="weight"):
    distances = {node: float('inf') for node in G.nodes}
    distances[source] = 0
    previous_nodes = {node: None for node in G.nodes}
    pq = [(0, source)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == target:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return path[::-1]

        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get(weight, 1)
            new_distance = current_distance + edge_weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))

    raise nx.NetworkXNoPath(f"Não há caminho entre {source} e {target}.")


def dijkstra_path_length(G, source, target, weight="weight"):
    """Retorna o comprimento do caminho mais curto entre source e target usando Dijkstra."""
    # Dicionário para armazenar as menores distâncias
    distances = {node: float('inf') for node in G.nodes}
    distances[source] = 0

    # Fila de prioridade (heap) para explorar os nós em ordem crescente de distância
    pq = [(0, source)]  # (distância acumulada, nó atual)

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Se chegamos ao destino, retornamos a menor distância encontrada
        if current_node == target:
            return current_distance

        # Explorar os vizinhos
        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get(weight, 1)  # Pega o peso da aresta
            new_distance = current_distance + edge_weight

            # Atualiza a menor distância conhecida até o vizinho
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))

    raise nx.NetworkXNoPath(f"Não há caminho entre {source} e {target}.")


def animate_chinese_postman(G, solution):
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = nx.get_edge_attributes(G, 'weight')

    def update(frame):
        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        edges_to_draw = solution[:frame + 1]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color='red', width=2)
        plt.title("Passo {} da execução".format(frame + 1))

    ani = animation.FuncAnimation(fig, update, frames=len(solution), interval=1000, repeat=False)
    plt.show()


def chinese_postman(G):
    if nx.is_eulerian(G):
        circuit = list(nx.eulerian_circuit(G))
        total_weight = sum(G[u][v]['weight'] for u, v in circuit)
        return circuit, total_weight

    odd_degree_nodes = [node for node in G.nodes if G.degree[node] % 2 == 1]
    min_pairs = None
    min_cost = float('inf')

    for pairs in permutations(odd_degree_nodes, len(odd_degree_nodes)):
        pairs = [(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)]
        cost = sum(dijkstra_path_length(G, u, v, weight='weight') for u, v in pairs)
        if cost < min_cost:
            min_cost = cost
            min_pairs = pairs

    for u, v in min_pairs:
        path = dijkstra_path(G, u, v)
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1], weight=G[path[i]][path[i + 1]]['weight'])

    circuit = list(nx.eulerian_circuit(G))
    total_weight = sum(G[u][v]['weight'] for u, v in circuit)
    return circuit, total_weight

# Criar e salvar dois grafos Eulerianos
#G1 = gerar_grafo_euleriano(nos=5, arestas=8)
#G2 = gerar_grafo_euleriano(nos=6, arestas=10)
#salvar_grafo_csv(G1, "grafo1.csv")
#salvar_grafo_csv(G2, "grafo2.csv")

# Exemplo de uso
graph = read_graph_from_csv("grafo.csv")
solution, total_weight = chinese_postman(graph)
print("Caminho do carteiro chinês:", solution)
print("Soma dos pesos do caminho:", total_weight)
plot_graph(graph)
animate_chinese_postman(graph, solution)
