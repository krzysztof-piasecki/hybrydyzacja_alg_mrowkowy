import random

def generate_dna_sequence(length):
    nucleotides = ['A', 'G', 'C', 'T']
    return ''.join(random.choice(nucleotides) for _ in range(length))

def introduce_errors(sequence, num_errors, k):
    sequence = list(sequence)
    for _ in range(num_errors):
        index = random.randint(0, len(sequence) - k)
        new_subsequence = generate_dna_sequence(k)
        while new_subsequence == sequence[index:index+k]:
            new_subsequence = generate_dna_sequence(k)
        sequence[index:index+k] = new_subsequence
    return ''.join(sequence)

k = 10
test_set = []
for _ in range(60):
    dna_length = 70
    num_errors = random.randint(1, 5)
    dna_sequence = generate_dna_sequence(dna_length * k)
    dna_sequence_with_errors = introduce_errors(dna_sequence, num_errors, k)
    test_set.append(dna_sequence_with_errors)

with open('dna_sequences.txt', 'w') as f:
    for sequence in test_set:
        f.write(sequence + '\n')
