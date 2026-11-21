import pandas as pd
from Levenshtein import distance
import numpy as np
from tqdm import tqdm
ALL_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def generate_unique_mutations(reference_sequences, num_mutations=1000, max_mutation_distance=0.25, all_amino_acids=ALL_AMINO_ACIDS, distance_func=distance):
    """
    Generate unique mutations for each reference sequence within a maximum distance threshold.

    Args:
        reference_sequences (list): List of reference sequences to mutate
        num_mutations (int): Number of unique mutations to generate per sequence
        max_mutation_distance (float): Maximum allowed distance ratio between mutation and reference
        all_amino_acids (list): List of valid amino acids for substitution
        distance_func (callable): Function to calculate distance between sequences

    Returns:
        list: List of lists containing unique mutations for each reference sequence
    """
    all_mutations = []
    for sequence in tqdm(reference_sequences):
        unique_mutations = set()
        attempts = 0
        max_attempts = num_mutations * 100
        while len(unique_mutations) < num_mutations and attempts < max_attempts:
            mutation = list(sequence)
            valid_mutation = False
            num_positions = np.random.randint(1, max(2, int(len(sequence) * max_mutation_distance)) + 1)
            mutated_positions = set()
            for _ in range(num_positions):
                available_positions = set(range(len(mutation))) - mutated_positions
                if not available_positions:
                    break
                mutation_position = np.random.choice(list(available_positions))
                mutated_positions.add(mutation_position)
                valid_choices = [aa for aa in all_amino_acids if aa != mutation[mutation_position]]
                if valid_choices:
                    mutation[mutation_position] = np.random.choice(valid_choices)
            mutation_str = ''.join(mutation)
            if distance_func(mutation_str, sequence) / len(sequence) <= max_mutation_distance and mutation_str not in unique_mutations and (mutation_str != sequence):
                unique_mutations.add(mutation_str)
                valid_mutation = True
            attempts += 1
        mutations_list = list(unique_mutations)
        if len(mutations_list) < num_mutations:
            print(f'Warning: Could only generate {len(mutations_list)} unique mutations for sequence {sequence}')
        all_mutations.append(mutations_list)
    return all_mutations

def main():
    from reference_peptides.all_1100_seeds import REFERENCE_SEQUENCE
    reference_sequences = REFERENCE_SEQUENCE[900:1000]
    unique_mutations = generate_unique_mutations(reference_sequences, num_mutations=1000)
    all_mutations_flatten = [x for xs in unique_mutations for x in xs]
    pd.DataFrame(all_mutations_flatten).to_csv('generated_inits/test_random_seed900_999_1k.csv', index=False, header=False)
if __name__ == '__main__':
    main()
