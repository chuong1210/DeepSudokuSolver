import time
import numpy as np
import matplotlib.pyplot as plt
from sudokuSolverDiversity import solveAStar, solveBFS, solveDFS, solveIDS
from DiTruyen import solveGA

def benchmark_algorithms(puzzle, algorithms, num_runs=5):
    results = {}
    for name, func in algorithms.items():
        print(f"Benchmarking {name}...")
        print(f"Type of func: {type(func)}")  # Add this line to debug
        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            func(np.copy(puzzle))  # Call the function
            end_time = time.time()
            total_time += (end_time - start_time)
        avg_time = total_time / num_runs
        results[name] = avg_time
    return results


def create_bar_chart(results, difficulty_level):
    algorithms = list(results.keys())
    times = list(results.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(algorithms, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.title(f'Sudoku Solving Algorithm Performance - {difficulty_level} Difficulty')
    plt.xlabel('Algorithms')
    plt.ylabel('Average Solving Time (seconds)')
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}s',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'sudoku_algorithm_performance_{difficulty_level}.png', dpi=300, bbox_inches='tight')
    plt.close()
puzzles = {
    "Easy": [
        np.array([list(map(int, "060209000000030010100600009420500090005302860083100024870906035340050270206073001"))]).reshape(9, 9),
        np.array([list(map(int, "920800040001340006403056800210064500004500000605102470008020013040010000097603204"))]).reshape(9, 9),
    ],
    "Medium": [
        np.array([list(map(int, "000079000000008210900162034003706800710005906500800000270910500000000000836000000"))]).reshape(9, 9),
        np.array([list(map(int, "685030407000800020010400500090300005040000600508004030926078300800000000003000019"))]).reshape(9, 9),
    ],
    "Hard": [
        np.array([list(map(int, "000013000000680010709000080000045001005006300340000000500009000070062590020000004"))]).reshape(9, 9),
        np.array([list(map(int, "006207000000005600100800205040008020290000000600012490000000000037000019005030000"))]).reshape(9, 9),
    ],
        "Expert": [
        np.array([list(map(int, "009800000500072000000000013090000002010396000700000000000400308105000400070020000"))]).reshape(9, 9),
        np.array([list(map(int, "038000400200005000000070000004000602000000001060800040080406100075000090006050080"))]).reshape(9, 9),
    ]
}

algorithms = {
    'A*': solveAStar,
    'BFS': solveBFS,
    'DFS': solveDFS,
    'IDS': solveIDS,
    'GA': solveGA
}

for difficulty, puzzle_set in puzzles.items():
    if(difficulty == "Easy"):
        continue
    if(difficulty == "Medium"):
        continue
    if(difficulty == "Hard"):
        continue
    print(f"\nBenchmarking Sudoku solving algorithms for {difficulty} puzzles...")
    results = {}
    for puzzle in puzzle_set:
        print(f"\nSolving puzzle: {puzzle}")
        results.update(benchmark_algorithms(puzzle, algorithms))

    print("\nResults:")
    for name, time in results.items():
        print(f"{name}: {time:.4f} seconds")

    print("\nCreating bar chart...")
    create_bar_chart(results, difficulty)
    print(f"Bar chart saved as 'sudoku_algorithm_performance_{difficulty}.png'")
