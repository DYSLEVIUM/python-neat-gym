import multiprocessing
import os
import pickle

import gymnasium as gym
import neat
import numpy as np

runs_per_net = 2

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make("CartPole-v1")
        # env = gym.make("LunarLander-v2")

        fitness = 0.0
        observation, info = env.reset()
        for _ in range(10000):
            action = np.argmax(net.activate(observation))

            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward

            if terminated or truncated:
                break

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open("winner", "wb") as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == "__main__":
    run()
