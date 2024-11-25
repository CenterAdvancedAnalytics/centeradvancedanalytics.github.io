## Genetic Algorithms
*Genetic algorithms are used in the world of artificial intelligence as practical tools for solving nonlinear optimization problems that operate in a very large search space. They are easy to explain, simple to implement, and extremely useful for solving real-world practical problems. Here, we explain how they work with two examples in Python.*

### 1. Basic Concepts
An optimization problem, whatever it may be, is formulated by considering the following components: i) **an objective function**, which is to be maximized or minimized; ii) a set of **control variables**, whose selection of values allows achieving the optimum of the objective function; and iii) a set of constraints on the control variables. The set of values that the control variables can take without violating any constraints is known as the **feasible set**.

One way to solve any optimization problem is brute force. That is, by testing candidate solutions in the feasible set and selecting the one that achieves the best performance when evaluated against the objective function.

**Example 1.** *Consider the problem of minimizing the function $(x − 3)^2 + (y − 7)^2$ for positive values of $x$ and $y$ such that $x + y \leq 10$. The objective function is $f(x, y) = (x − 3)^2 + (y − 7)^2$, the control variables are $x$ and $y$, and the constraints on these variables are $x \geq 0$, $y \geq 0$, and $x + y \leq 10$.*

In mathematical terms, this problem is nothing more than

$$
(x^*, y^*) = \argmax_{x,y}\left\{1,2\right\}
$$ 
*In this case, the objective function is not linear, and finding its solution is not immediate for someone without basic knowledge of calculus. Under these circumstances, it is reasonable to use the brute force method. To do this, we randomly select some values for the pair (x, y), say {(1, 6), (5, 3), (−1, 9)}, and check if they belong to the feasible set. In this case, the first two candidates meet the constraints while the third one does not, so the pair (−1, 9) is discarded, and thus we define S = {(1, 6), (5, 3)} as the set of feasible candidate solutions. Finally, we evaluate each solution in S to see which one gives us the smallest value. In this case, f(1, 6) = 5 is the best solution, since f(5, 3) = 20 > 5. Of course, nothing guarantees that this solution corresponds to the global minimum. In fact, the argument that minimizes f(x, y) in the feasible space is (3, 7).*

A genetic algorithm is an intelligent way to apply the brute force method, using a “bio-inspired” strategy to probe values. In this sense, it is an algorithm that recursively explores the feasible space, using information from the solutions obtained in the previous step to generate new solutions. It is bio-inspired because it resembles the procreation mechanism of biological organisms. From this perspective, the parents (previous solutions) produce offspring (current solutions) under certain crossover rules (combination of solutions) and mutation (alterations of the current solutions).

Next, we will delve deeper into the optimization process using the Python programming language. We will see how to build a genetic algorithm step by step, defining the key functions and the necessary parameters for its execution. To structure the presentation, we will provide two examples.

### 2. An Algebraic Function
#### 2.1 Problem Setup
We start the algorithm by creating a class and initializing its key parameters. The constructor receives the bounds where the quadratic function will be minimized (**bounds**), the population size (**population_size**), the number of generations to be executed (**num_generations**), the crossover rate (**crossover_rate**), and the mutation rate (**mutation_rate**). Finally, two lists are created to store the best fitness value and the best individual of each generation.
```python
def __init__(self, bounds, population_size=100, num_generations=100,
crossover_rate=0.9, mutation_rate=0.1):
  self.bounds = bounds
  self.population_size = population_size
  self.num_generations = num_generations
  self.crossover_rate = crossover_rate
  self.mutation_rate = mutation_rate
  self.best_fitness_per_generation = []
  self.best_individual_per_generation = []
```

#### 2.2 Population Initialization
Next, the function **create_population** is responsible for initializing the population. Each individual is a random decimal number within the established bounds, and the population is represented as a list of individuals.
```python
def create_population(self):
  population = []
  for _ in range(self.population_size):
  individual = random.uniform(self.bounds[0], self.bounds[1])
  population.append(individual)
  return population
```

#### 2.3 Parent Selection
The **select_parents** method chooses pairs of parents for the next generation of solutions. Selection is carried out through a tournament where a subset of 2 elements in the population is determined, and the one with the better fitness is selected as a parent. This process is repeated until selecting as many pairs of parents as half the population size.
```python
def select_parents(self, population):
  parents = []
  for _ in range(self.population_size // 2):
  parent1 = self.tournament_selection(population)
  parent2 = self.tournament_selection(population)
  parents.append((parent1, parent2))
  return parents

def tournament_selection(self, population):
  competitors = random.sample(population, 2)
  return min(competitors, key=self.fitness_function)
```
It is important to note that there are various methods for parent selection, each suitable for different conditions and characteristics of the problem. The choice of selection method (**roulette** selection, **rank-based** selection, **elitist** selection) can significantly influence the efficiency and effectiveness of the algorithm. Therefore, it is advisable to experiment with different methods to find the most suitable one for each particular situation.

#### 2.4 Crossover and Mutation
Another set of exploration rules in the algorithm are provided by its crossover and mutation methods. The crossover function utilizes the characteristics of two parents to generate offspring only if the crossover probability is met. In our implementation, since each individual is represented by a real number, the average of the parents is calculated to generate a child.

On the other hand, the mutate function is responsible for altering individuals only if the mutation probability is met. If this is the case, a fraction of a random value within the search space limits is added. If not, the individual remains unchanged.
```python
def crossover(self, parent1, parent2):
  if random.random() < self.crossover_rate:
  child1 = (parent1 + parent2)/2
  child2 = (parent1 + parent2)/2
  else:
  child1 = parent1
  child2 = parent2
  return child1, child2

def mutate(self, individual):
  mutated_individual = individual
  if random.random() < self.mutation_rate:
  mutated_individual = mutated_individual +
  0.1 * random.uniform(self.bounds[0], self.bounds[1])
  else:
  mutated_individual = mutated_individual
  return mutated_individual
```
Both methods have exploratory characteristics, employing stochasticity that allows escaping local optima. Crossover mixes genetic information from parents to produce new individuals, while mutation introduces random variations in individuals, ensuring genetic diversity in the population. This combination of crossover and mutation helps to prevent premature convergence and enhances the algorithm’s ability to explore the solution space in search of the global optimum.

#### 2.5 Survivor Selection
We have mentioned earlier that the selection of optimal individuals takes place after measuring their fitness level. Now, we will examine this process in detail. First, the **fitness_function** calculates an individual’s fitness as the square of its value; the lower the individual’s value, the higher its fitness. Individual selection is performed with the **select_survivors** method, in which the best solutions are chosen as survivors.
```python
def select_survivors(self, population, num_survivors):
  sorted_population = sorted(population, key=self.fitness_function)
  return sorted_population[:num_survivors]
```
It is important to mention that, although there are various methods for survivor selection in genetic algorithms, in this implementation, we have chosen to select the best individuals based on their fitness. Other methods, such as **niche selection** or **age selection**, may be more appropriate depending on the characteristics of the problem and may influence the diversity and convergence of the algorithm.

#### 2.6 Execution
Now is the time to gather all the previously described functions and execute the genetic algorithm. The **evolve** method generates a population using **create_population**. During each generation, it selects parents through **select_parents**, combines parents to generate new solutions, and alters individuals with crossover and mutate. Finally, it selects the fittest individuals with the **select_survivors** function.
```python
def evolve(self):
  population = self.create_population()
  for generation in range(self.num_generations):
  parents = self.select_parents(population)
  children = []
  for parent1, parent2 in parents:
  child1, child2 = self.crossover(parent1, parent2)
  children.extend([child1, child2])
  for i in range(len(population)):
  population[i] = self.mutate(population[i])
  population = self.select_survivors(population + children, self.population_size)
  best_individual = population[0]
  best_fitness = self.fitness_function(best_individual)
  self.best_fitness_per_generation.append(best_fitness)
  self.best_individual_per_generation.append(best_individual)
  
  print(f"Generation {generation+1:3d}, Best fitness: {best_fitness}")
  return self.best_fitness_per_generation, self.best_individual_per_generation
```
To test the proposed genetic algorithm, we conducted a simulation using values ranging from -500 to 500, a population of 50 individuals, 100 generations with a reproduction and mutation rate set at 0.5. The results obtained are observed in the evolution of the fitness function of the best individual in each generation:

![Genetic Algorithm Convergence](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-29-genetic-algorithms/fitness_function_evolution.png "Genetic Algorithm Convergence")

It can be observed that the genetic algorithm converges around the first 10 generations and finds the minimum of the quadratic function at 0.

#### 3. The Traveling Salesman
Another example of applying the genetic algorithm is in the traveling salesman problem (TSP). This is a combinatorial optimization problem in which the goal is to find the shortest route that allows a salesman to visit a list of cities, passing through each one exactly once and returning to the starting city. The objective is to minimize the total distance traveled. The implementation differs from the previous example in how an individual is modeled, crossover, and mutation, as now lists of city combinations are employed where the order of visitation is crucial.

Consider a map of 4 cities, where the numbers indicated on the arcs denote the distances between the cities.

![Traveling Salesman](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-29-genetic-algorithms/traveling_salesman.png "Traveling Salesman")
 
The search space is modeled using the following distance matrix:
```python
distance_matrix = [[0, 20, 42, 35],
                   [20, 0, 30, 34],
                   [42, 30, 0, 12],
                   [35, 34, 12, 0]]
```
The main difference between the genetic algorithm for this problem and the previous one lies in the characterization of the individual. In the previous case, it is defined as a real number, while in the traveling salesman problem, it is a list indicating the order in which the cities are visited.
```python
def __init__(self, distance_matrix, population_size=100, num_generations=100,
crossover_rate=0.9, mutation_rate=0.1):
  self.distance_matrix = distance_matrix
  self.num_cities = len(distance_matrix)
  self.population_size = population_size
  self.num_generations = num_generations
  self.crossover_rate = crossover_rate
  self.mutation_rate = mutation_rate
  self.best_fitness_per_generation = []
  self.best_individual_per_generation = []
```
The population initialization is done by permutations of the order of city visits.
```python
def create_population(self):
  starting_point = list(np.random.permutation(self.num_cities))
  population = []
  for _ in range(self.population_size):
  individual = starting_point.copy()
  population.append(individual)
  return population
```
Crossover is done using the one-point type, which involves selecting a random index point from the individual’s list and filling the child’s list from the beginning of the parent’s list up to the point, and using the remaining cities from the second parent while maintaining their order. This is possible because lists are used, and the definition of crossover is fulfilled by employing components from both parents to form the child.
```python
def crossover(self, parent1, parent2):
  if random.random() < self.crossover_rate:
  crossover_point = random.randint(1, self.num_cities - 1)
  child1 = self.order_crossover(parent1, parent2, crossover_point)
  child2 = self.order_crossover(parent2, parent1, crossover_point)
  else:
  child1 = parent1
  child2 = parent2
  return child1, child2

def order_crossover(self, parent1, parent2, crossover_point):
  child = [-1] * self.num_cities
  child[:crossover_point] = parent1[:crossover_point]
  current_pos = crossover_point
  for city in parent2:
  if city not in child:
  child[current_pos] = city
  current_pos += 1
  return child
```
The swap mutation type is selected, in which two random positions in the individual’s list are exchanged. This makes sense only if the order in the list is important, which is the case for the described problem where the visitation order influences the travel time.
```python
def mutate(self, individual):
  if random.random() < self.mutation_rate:
  i, j = random.sample(range(self.num_cities), 2)
  individual[i], individual[j] = individual[j], individual[i]
  return individual
```
The fitness function calculates the total travel time from the first point to the last point and the return time to the first point. The provided distance matrix in the constructor is used, and through indices, the distances from going from one point to another in the individual’s list are accessed. This is done for each element of the list, and the total time is accumulated.
```python
def fitness_function(self, individual):
  total_distance = 0
  for i in range(self.num_cities - 1):
  total_distance += self.distance_matrix[individual[i]][individual[i+1]]
  total_distance += self.distance_matrix[individual[-1]][individual[0]]
  return total_distance
```
To test the algorithm, the previously presented matrix is used, and the algorithm is executed with a population size of 10, mutation and crossover rate of 0.2, and for 10 generations. The evolution of fitness is as follows:

![Genetic Algorithm Evolution](https://raw.githubusercontent.com/CenterAdvancedAnalytics/centeradvancedanalytics.github.io/refs/heads/main/_posts/images/2024-05-29-genetic-algorithms/fitness_function_evolution_2.png "Genetic Algorithm Evolution")
 
The solution obtained is a travel time of 97 hours with the visiting order: [C, B, A, D, C]. For this problem, there are multiple solutions as different permutations can generate the same minimum time, depending on where one starts.

#### 4. Final Comments
Despite their effectiveness and adaptability, genetic algorithms have certain limitations. Among them are their high computational demand, the possibility of premature convergence to suboptimal solutions, and the need for careful parameter tuning to obtain good results. Additionally, genetic algorithms may be less efficient for certain types of specific problems. As alternatives, there are other heuristic methods that can also effectively address large and nonlinear problems. These include Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Tabu Search, and Simulated Annealing (SA) algorithms. Each of these methods has its own strengths and weaknesses, and the choice of the most suitable algorithm depends on the specific characteristics of the problem and the available resources.

#### 5. References
[1] D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning, Addison-Wesley Longman Publishing Co., Inc., 1989.

[2] J. Kennedy and R. Eberhart, “Particle swarm optimization”, Proceedings of ICNN’95-International Conference on Neural Networks, vol. 4, pp. 1942–1948, IEEE, 1995.

[3] M. Dorigo and T. Stützle Ant Colony Optimization, MIT Press, 2004.


