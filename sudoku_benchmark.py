import time
import numpy as np
import matplotlib.pyplot as plt
from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS, solveIDS, solveGreedy
from DiTruyen import solveGA

def benchmark_algorithms(puzzle, algorithms, num_runs=5):
    results = {}
    for name, func in algorithms.items():
        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            func(np.copy(puzzle))
            end_time = time.time()
            total_time += (end_time - start_time)
        avg_time = total_time / num_runs
        results[name] = avg_time
    return results

def create_bar_chart(results):
    algorithms = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(algorithms, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.title('Sudoku Solving Algorithm Performance Comparison')
    plt.xlabel('Algorithms')
    plt.ylabel('Average Solving Time (seconds)')
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('sudoku_algorithm_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example Sudoku puzzle (you can replace this with your own puzzle)
puzzle = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])

algorithms = {
    'A*': solveAStar,
    'BFS': solveBFS,
    'DFS': solveDFS,
    'IDS': solveIDS,
    'Greedy': solveGreedy,
    'GA': solveGA
}

print("Benchmarking Sudoku solving algorithms...")
results = benchmark_algorithms(puzzle, algorithms)

print("\nResults:")
for name, time in results.items():
    print(f"{name}: {time:.4f} seconds")

print("\nCreating bar chart...")
create_bar_chart(results)
print("Bar chart saved as 'sudoku_algorithm_performance.png'")