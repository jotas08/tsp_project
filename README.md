# TSP_Project
This project focuses on solving the **Traveling Salesman Problem (TSP)** using various metaheuristic algorithms. The **TSP** is a classic combinatorial optimization problem where the goal is to find the shortest possible tour that visits a set of cities exactly once and returns to the starting point.

The dataset used for this project comes from the **TSPLIB95**, a library of sample instances for the TSP and related problems. We specifically selected the **att48.tsp** instance for testing, which contains 48 cities with real-world coordinates.

The **TSPLIB95** dataset can be found at the following URL:
[TSPLIB95 - TSP Instances](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)

In this project, we implement multiple metaheuristics to find optimal or near-optimal solutions for the **TSP**, including:

- Randomized Multi-Start (RMS)
- Iterated Local Search (ILS)
- Variable Neighborhood Search (VNS)
- Simulated Annealing
- Tabu Search
- Greedy Randomized Adaptive Search Procedure (GRASP)

Also some methods, including:

- Greedy Method
- Random Method

The att48.tsp instance provides a real-world challenge, and through these algorithms, we demonstrate different approaches to solving the problem efficiently.

## Motivation

This project was created as part of the **Combinatorial Optimization and Metaheuristics** course at the **Federal University of Ceará**, in the **Industrial Mathematics** program. The goal of the project is to fulfill the requirements of the course and demonstrate proficiency in solving complex optimization problems using metaheuristic techniques.

Specifically, this project addresses the first 13 tasks outlined in the course assignment, providing a comprehensive solution to the **Traveling Salesman Problem (TSP)** using several advanced algorithms:

1. Choose the problem
2. Implement a method to read instances from a benchmark file and store them in a TSP data structure: TSPLIB
3. Implement an exact solution for small instances
4. Implement a random solution constructor
5. Implement a greedy solution constructor
6. Implement a local search method (single neighborhood)
7. Implement a VND (Variable Neighborhood Descent) method (multiple neighborhoods)
8. Implement a random multi-start method
9. Implement an iterated local search method
10. Implement a variable neighborhood search method
11. Implement a simulated annealing method
12. Implement a tabu search method
13. Implement a GRASP method (Greedy Randomized Adaptive Search Procedure)
14. ~~Implement a p-metaheuristic studied in class (Estimation of Distribution Algorithm, Genetic Algorithm, Scatter Search)~~

The 14th task will be implemented later.

## Usage

To use this project, you will need a **TSP instance file** for reading, such as those available from the **TSPLIB95** dataset. For example, you can load the **att48.tsp** instance as follows:

> tsp = TSP(filename='ALL_tsp/att48.tsp')

Once you have instantiated the **TSP** class with the desired instance file, you can apply various local search methods and constructive heuristics by instantiating their respective classes. For instance, you can use the *Simulated Annealing*, *GRASP*, or any other implemented algorithm.

For local search and constructive heuristic methods, you instantiate the respective class and apply it to the TSP problem. Each algorithm class is designed to optimize the *TSP* instance.

Thanks to the `__str__` *method* implemented in the classes, you can easily display the results directly by printing the solution object. Additionally, you can plot the tour using the `plot_tsp` method by passing a string as a description and the *objective cost* of the solution, as follows:

> tsp.plot_tsp('Greedy Solution', sol.objective)

This provides a visual representation of the tour and its total cost.

Moreover, the use of the **TSP_Solution** class allows for flexible manipulation of the problem, making it easy to apply various heuristics, optimize the tour, and evaluate results in a structured way.

### Metaheuristics Parameters

Each metaheuristic implemented in this project comes with specific parameters that control their behavior and performance. Below is a general explanation of the most commonly used parameters:

1. **max_tries**: This parameter sets the maximum number of iterations or attempts before the algorithm terminates. It's used in most metaheuristics to define a stopping condition when no further improvements are found.

2. **first_imp (first improvement)**: When set to `True`, the algorithm will stop the search within a neighborhood at the first improvement found. This makes the search faster but might not always find the best improvement. When set to `False` the algorithm will explore the entire neighborhood to find the best possible improvement.

3. **K**: In methods like **GRASP**, this defines the number of candidates to consider in the greedy randomized construction phase. A higher `K` adds more diversity but can also slow down the process.

4. **T0 (initial temperature) and alpha (cooling factor)**: These parameters are used in **Simulated Annealing (SA)**. `T0` controls the initial temperature, determining how frequently worse solutions are accepted at the start. `alpha` is the cooling factor that reduces the temperature gradually, making the algorithm less likely to accept worse solutions over time.

5. **tenure**: In **Tabu Search**, this parameter controls how long a move or solution is kept in the **tabu list** (prohibited list). A higher tenure value means that moves are tabu for more iterations, helping the algorithm explore new regions of the solution space.

6. **k_max**: In **Variable Neighborhood Search (VNS)**, this parameter defines the maximum size of the neighborhoods explored. It determines how far the algorithm can "jump" from the current solution to explore other neighborhoods.

# Contact

If you have any questions or suggestions regarding this project, feel free to contact me:

<a href="mailto:jotasu.drive@gmail.com">
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail" width="30" height="30">
</a>
<a href="https://www.linkedin.com/in/jo%C3%A3o-paulo-venancio-silva-6a71532a7/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="30" height="30">
</a>