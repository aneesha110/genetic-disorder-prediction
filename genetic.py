import pandas as pd
from sklearn.preprocessing import MinMaxScaler
train_data = pd.read_csv('normalized_data.csv')
scaler = MinMaxScaler()
#data_norm = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
#dropping target variables
train = train_data.drop(['Genetic Disorder', 'Disorder Subclass'], axis = 1)
#normalization



target = train_data[['Genetic Disorder', 'Disorder Subclass']]
target = target.copy()

# Convert "Genetic Disorder" column to category type and encode it
target.loc[:, "Genetic Disorder"] = target.loc[:, "Genetic Disorder"].astype('category').cat.codes

# Convert "Disorder Subclass" column to category type and encode it
target.loc[:, "Disorder Subclass"] = target.loc[:, "Disorder Subclass"].astype('category').cat.codes
X = train_data.to_numpy()
y = target.to_numpy()

#split dataset: 80% for train set and 20% for test set
num_of_rows = int(len(X) * 0.8)
X_train = X[:num_of_rows]
X_test = X[num_of_rows:]

#splitting target variables
Y_train = y[:num_of_rows]
Y_train_1 = Y_train[:, -2] #for Genetic Disorder
Y_train_2 = Y_train[:, -1] #for Disorder Subclass

Y_test = y[num_of_rows:]
Y_test_1 = Y_test[:, -2]
Y_test_2 = Y_test[:, -1]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import numpy as np
np.random.seed(42)  # Set random seed for reproducibility


# Generate a random dataset for demonstration
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define hyperparameters for genetic algorithm
population_size = 50
num_generations = 50
crossover_rate = 0.8
mutation_rate = 0.1
tournament_size = 3

# Function to initialize the population with random chromosomes
def initialize_population(population_size, chromosome_length):
    return np.random.randint(2, size=(population_size, chromosome_length))

# Function to evaluate the fitness of each chromosome using XGBoost classifier
def calculate_fitness(chromosomes, X_train, Y_train, X_test, Y_test):
    fitness = []

    for chromosome in chromosomes:
        selected_features = [i for i in range(len(chromosome)) if chromosome[i] == 1]
        if not selected_features:
            fitness.append(0)  # Avoid all-zero chromosomes
        else:
            # Train XGBoost model for each target variable
            clf1 = XGBClassifier(random_state=42)
            clf1.fit(X_train[:, selected_features], Y_train[:, 0])  # Train for Genetic Disorder

            clf2 = XGBClassifier(random_state=42)
            clf2.fit(X_train[:, selected_features], Y_train[:, 1])  # Train for Disorder Subclass

            y_pred1 = clf1.predict(X_test[:, selected_features])
            y_pred2 = clf2.predict(X_test[:, selected_features])

            # Evaluate accuracy for each target variable
            accuracy1 = accuracy_score(Y_test[:, 0], y_pred1)
            accuracy2 = accuracy_score(Y_test[:, 1], y_pred2)

            # Calculate average accuracy as fitness
            fitness.append((accuracy1 + accuracy2) / 2)

    return np.array(fitness)

# Function to perform crossover operation
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Function to perform mutation operation
def mutate(chromosome, mutation_rate):
    mutation_mask = np.random.rand(len(chromosome)) < mutation_rate
    chromosome[mutation_mask] = 1 - chromosome[mutation_mask]
    return chromosome

# Function to select the next generation based on tournament selection
def select_next_generation(chromosomes, fitness, tournament_size):
    selected_indices = []

    for _ in range(len(chromosomes)):
        tournament_candidates = np.random.choice(len(chromosomes), size=tournament_size, replace=False)
        tournament_fitness = fitness[tournament_candidates]
        winner_index = tournament_candidates[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)

    return chromosomes[selected_indices]

# Main genetic algorithm function
def genetic_algorithm(X_train, Y_train, X_test, Y_test, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.1, tournament_size=3):
    chromosome_length = X_train.shape[1]
    population = initialize_population(population_size, chromosome_length)

    for generation in range(generations):
        fitness = calculate_fitness(population, X_train, Y_train, X_test, Y_test)

        # Select parents using tournament selection
        parents = select_next_generation(population, fitness, tournament_size)

        # Perform crossover
        crossover_indices = np.random.rand(len(parents)) < crossover_rate
        crossover_parents = parents[crossover_indices]

        # Ensure that crossover_parents has an even length
        if len(crossover_parents) % 2 != 0:
            crossover_parents = crossover_parents[:-1]

        crossover_children = np.empty_like(crossover_parents)

        for i in range(0, len(crossover_parents), 2):
            crossover_children[i], crossover_children[i + 1] = crossover(crossover_parents[i], crossover_parents[i + 1])

        # Perform mutation
        mutation_indices = np.random.rand(len(crossover_children)) < mutation_rate
        crossover_children[mutation_indices] = mutate(crossover_children[mutation_indices], mutation_rate)

        # Replace old population with new population
        population[:len(crossover_children)] = crossover_children

    # Get the best chromosome from the final population
    best_chromosome = population[np.argmax(fitness)]

    return best_chromosome

# Example usage
best_chromosome = genetic_algorithm(X_train, np.column_stack((Y_train_1, Y_train_2)), X_test, np.column_stack((Y_test_1, Y_test_2)))

# Display selected features based on the best chromosome
selected_features_genetic_algorithm = [i for i in range(len(best_chromosome)) if best_chromosome[i] == 1]
print("Selected Features based on Genetic Algorithm:")
print(selected_features_genetic_algorithm)

# Train final model using selected features
clf1 = XGBClassifier(random_state=42)
clf1.fit(X_train[:, selected_features_genetic_algorithm], Y_train_1)

clf2 = XGBClassifier(random_state=42)
clf2.fit(X_train[:, selected_features_genetic_algorithm], Y_train_2)

# Evaluate accuracy using the selected features
y_pred1 = clf1.predict(X_test[:, selected_features_genetic_algorithm])
accuracy_genetic_algorithm1 = accuracy_score(Y_test_1, y_pred1)

y_pred2 = clf2.predict(X_test[:, selected_features_genetic_algorithm])
accuracy_genetic_algorithm2 = accuracy_score(Y_test_2, y_pred2)

print("Accuracy after Genetic Algorithm (Genetic Disorder):", accuracy_genetic_algorithm2)
