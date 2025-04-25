import random
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms import isomorphism

class Node:
    __slots__ = ['data', 'left', 'right']
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
        self._size = 0

    def get_size(self):
        return self._size

    def create_random_tree(self, num_nodes, none_probability=0.2):
        start_time = time.time()
        if num_nodes <= 0:
            raise ValueError("Количество узлов должно быть положительным")

        values = [random.randint(1, num_nodes) if random.random() >= none_probability else None for _ in range(num_nodes)]
        if values[0] is None:
            values[0] = random.randint(1, num_nodes)

        self.root = Node(values[0])
        self._size = 1
        nodes_queue = deque([self.root])
        current_index = 1

        while nodes_queue and current_index < num_nodes:
            current_node = nodes_queue.popleft()
            if current_node is None:
                current_index += 2
                continue

            if current_index < num_nodes:
                if values[current_index] is not None:
                    current_node.left = Node(values[current_index])
                    self._size += 1
                    nodes_queue.append(current_node.left)
                else:
                    nodes_queue.append(None)
                current_index += 1

            if current_index < num_nodes:
                if values[current_index] is not None:
                    current_node.right = Node(values[current_index])
                    self._size += 1
                    nodes_queue.append(current_node.right)
                else:
                    nodes_queue.append(None)
                current_index += 1

        elapsed = (time.time() - start_time) * 1_000
        print(f"Дерево создано за {elapsed:.3f} мс")
        print(f"Количество узлов в дереве: {self._size}")
        self.save_to_file("generated_tree.txt")

    def visualize(self, title="Бинарное дерево", max_nodes=None):
        if self.root is None:
            print("Дерево пустое")
            return

        if max_nodes is None:
            max_nodes = self._size

        G = nx.DiGraph()
        pos = {}

        def _build_graph(node, x=0, y=0, dx=1, count=[0]):
            if node is None or count[0] >= max_nodes:
                return
            node_id = id(node)
            G.add_node(node_id, label=str(node.data))
            pos[node_id] = (x, -y)
            count[0] += 1
            if node.left:
                G.add_edge(node_id, id(node.left))
                _build_graph(node.left, x - dx, y + 1, dx / 2, count)
            if node.right:
                G.add_edge(node_id, id(node.right))
                _build_graph(node.right, x + dx, y + 1, dx / 2, count)

        _build_graph(self.root)

        node_colors = ['skyblue' if G.out_degree(n) else 'lightgreen' for n in G.nodes()]
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}

        plt.figure(figsize=(12, 8))
        nx.draw(
            G, pos,
            labels=labels,
            node_size=1500,
            node_color=node_colors,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7,
            width=2,
            arrows=False,
            edgecolors='black',
            style='dotted'
        )
        plt.title(title, fontsize=16, weight='bold')
        plt.show()

    def save_to_file(self, filename):
        if self.root is None:
            print("Нечего сохранять — дерево пустое")
            return
        with open(filename, "w", encoding="utf-8") as f:
            def _preorder(node):
                if node is None:
                    f.write("None\n")
                    return
                f.write(f"{node.data}\n")
                _preorder(node.left)
                _preorder(node.right)
            _preorder(self.root)
        print(f"Дерево сохранено в файл: {filename}")

    def load_from_file(self, filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            print(f"Файл '{filename}' не найден.")
            return

        def _build_tree(iterator):
            try:
                value = next(iterator)
            except StopIteration:
                return None
            if value == "None":
                return None
            node = Node(int(value))
            self._size += 1
            node.left = _build_tree(iterator)
            node.right = _build_tree(iterator)
            return node

        self._size = 0
        self.root = _build_tree(iter(lines))
        print(f"Дерево загружено из файла: {filename} ({self._size} узлов)")

    def to_networkx(self):
        G = nx.DiGraph()

        def add_edges(node):
            if node is None:
                return
            node_id = id(node)
            G.add_node(node_id, label=str(node.data))
            if node.left:
                G.add_edge(node_id, id(node.left))
                add_edges(node.left)
            if node.right:
                G.add_edge(node_id, id(node.right))
                add_edges(node.right)

        add_edges(self.root)
        return G

def visualize_subtree(subgraph, idx):
    pos = nx.nx_pydot.graphviz_layout(subgraph, prog="dot")
    labels = {n: subgraph.nodes[n]['label'] for n in subgraph.nodes()}
    node_colors = ['lightcoral' for _ in subgraph.nodes()]

    plt.figure(figsize=(6, 4))
    nx.draw(
        subgraph, pos,
        labels=labels,
        node_size=1500,
        node_color=node_colors,
        font_size=12,
        font_weight='bold',
        edge_color='gray',
        alpha=0.7,
        width=2,
        arrows=False,
        edgecolors='black',
        style='dotted'
    )
    plt.title(f"Совпадение #{idx}", fontsize=16, weight='bold')
    plt.show()

def find_structurally_matching_subtrees_networkx(main_tree: BinaryTree, pattern_tree: BinaryTree):
    main_graph = main_tree.to_networkx()
    pattern_graph = pattern_tree.to_networkx()

    GM = isomorphism.DiGraphMatcher(
        main_graph,
        pattern_graph,
        node_match=lambda n1, n2: True
    )

    unique_matches = set()
    final_matches = []

    for match in GM.subgraph_isomorphisms_iter():
        sorted_nodes = tuple(sorted(match.keys()))
        if sorted_nodes not in unique_matches:
            unique_matches.add(sorted_nodes)
            final_matches.append(match)

    print(f"Найдено {len(final_matches)} уникальных совпадающих подграфов (через NetworkX).")
    return final_matches

def manual_tree_creation_from_list():
    print("\nВведите значения узлов через запятую, используя 'None' для пропущенных узлов.")
    print("Пример: 1, 2, None, 3, 4")
    input_values = input("Введите список значений: ").strip()

    try:
        values = [None if val.strip().lower() == 'none' else int(val) for val in input_values.split(",")]
    except ValueError:
        print("Неверный ввод! Убедитесь, что вы ввели целые числа или 'None'.")
        return None

    tree = BinaryTree()
    if not values or values[0] is None:
        print("Невозможно создать дерево с пустым корнем.")
        return None

    tree.root = Node(values[0])
    tree._size = 1
    queue = deque([tree.root])
    index = 1

    while queue and index < len(values):
        current = queue.popleft()

        if index < len(values) and values[index] is not None:
            current.left = Node(values[index])
            tree._size += 1
            queue.append(current.left)
        index += 1

        if index < len(values) and values[index] is not None:
            current.right = Node(values[index])
            tree._size += 1
            queue.append(current.right)
        index += 1

    print(f"Дерево создано. Количество узлов: {tree._size}")
    return tree

def main():
    current_tree = None
    while True:
        print("\nМеню:")
        print("1. Загрузить дерево из файла")
        print("2. Сгенерировать случайное дерево")
        print("3. Создать дерево вручную")
        print("4. Показать текущее дерево")
        print("5. Найти поддеревья (NetworkX)")
        print("6. Сохранить дерево")
        print("7. Выход")

        choice = input("Выберите действие: ")

        if choice == '1':
            filename = input("Имя файла: ")
            current_tree = BinaryTree()
            current_tree.load_from_file(filename)

        elif choice == '2':
            num_nodes = int(input("Количество узлов: "))
            prob = float(input("Вероятность None (0-1): "))
            current_tree = BinaryTree()
            current_tree.create_random_tree(num_nodes, prob)

        elif choice == '3':
            current_tree = manual_tree_creation_from_list()

        elif choice == '4':
            if current_tree:
                current_tree.visualize()
            else:
                print("Дерево не загружено.")

        elif choice == '5':
            if not current_tree:
                print("Нет текущего дерева!")
                continue
            pattern_tree = manual_tree_creation_from_list()
            if not pattern_tree:
                continue

            start_time = time.time()
            matches = find_structurally_matching_subtrees_networkx(current_tree, pattern_tree)
            elapsed_time = (time.time() - start_time) * 1_000
            print(f"Поиск совпадающих поддеревьев выполнен за {elapsed_time:.3f} мс")

            if not matches:
                continue

            full_graph = current_tree.to_networkx()
            skip_visuals = False

            for idx, match in enumerate(matches, 1):
                labels = [full_graph.nodes[n]['label'] for n in match.keys()]
                print(f"{idx}. {', '.join(labels)}")

                if not skip_visuals:
                    while True:
                        user_input = input("Показать это поддерево? (y — да, n — нет, s — пропустить все): ").strip().lower()
                        if user_input == 'y':
                            subgraph_nodes = match.keys()
                            subgraph = full_graph.subgraph(subgraph_nodes).copy()
                            visualize_subtree(subgraph, idx)
                            break
                        elif user_input == 'n':
                            break
                        elif user_input == 's':
                            skip_visuals = True
                            break
                        else:
                            print("Введите 'y', 'n' или 's'.")

        elif choice == '6':
            if current_tree:
                filename = input("Имя файла: ")
                current_tree.save_to_file(filename)
            else:
                print("Нет дерева для сохранения.")

        elif choice == '7':
            print("Выход.")
            break

        else:
            print("Неверный выбор!")

if __name__ == "__main__":
    main()
