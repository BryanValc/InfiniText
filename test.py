import main

import random
import re
import time
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# We load the words from the file
with open('input/alice.txt', 'r') as file:
    words = file.read().split()

    for i in range(len(words)):
        words[i] = re.sub('[^a-z]', '', words[i].lower())

test_texts = []

# Generate samples of 3 to 5 words
for num_words in range(3, 6):
    samples = []
    for _ in range(100):
        start_index = random.randint(0, len(words) - num_words)
        sample = words[start_index:start_index + num_words]
        samples.append(' '.join(sample))
    test_texts.append(samples)

# Load the model
model = main.load_model('model.json')

# Initialize the lists of times for encoding and decoding
encoding_times = [[] for _ in range(3)]  # 5 because we are testing for 4 to 8 words
decoding_times = [[] for _ in range(3)]
compression_rates = [[] for _ in range(3)]

# For each test text we encode and decode it, and measure the times
success_count = 0
failure_count = 0

for i, samples in enumerate(test_texts):
    for test_text in samples:
        original_size = len(test_text) * 8
        compression_rate = original_size / 98
        compression_rates[i].append(compression_rate)

        # encoding
        start_time = time.time()
        bit_string_letters, bit_string_length, bit_word_count = main.encode_target_text(test_text)
        submodel = main.create_submodel(model, bit_string_letters, bit_string_length)
        bit_word_index = main.get_word_index(test_text, submodel)
        seed = main.find_matching_seed(test_text, submodel, int(bit_word_index, 2), int(bit_word_count, 2))
        encoding_times[i].append(time.time() - start_time)

        # Decodificación
        start_time = time.time()
        decoded_text = main.recreate_text(model, bit_string_letters, bit_string_length, bit_word_count, bit_word_index, seed)
        decoding_times[i].append(time.time() - start_time)

        # Assertion tests
        if test_text != decoded_text:
            print(f"Error: {test_text} != {decoded_text}")
            failure_count += 1
        else:
            print(f"Success: {test_text} == {decoded_text}")
            success_count += 1
        # Printing individual results
        print(f"Encoding(ms): {encoding_times[i][-1] * 1000:.2f}")
        print(f"Decoding(ms): {decoding_times[i][-1] * 1000:.2f}")
        print(f"Compression rate: {compression_rate:.2f}")

print(f"Success count: {success_count}")
print(f"Failure count: {failure_count}")
print(f"Global average encoding time: {np.mean(encoding_times):.2f}")
print(f"Global average decoding time: {np.mean(decoding_times):.2f}")
print(f"Global average compression rate: {np.mean(compression_rates):.2f}")

# Convertir las listas a arrays de numpy para facilitar los cálculos
encoding_times = np.array(encoding_times)
decoding_times = np.array(decoding_times)
compression_rates = np.array(compression_rates)

# Loop over each set of tests
for i in range(3):
    # Medidas de tendencia central
    mean_encoding = np.mean(encoding_times[i])
    mean_decoding = np.mean(decoding_times[i])
    mean_compression = np.mean(compression_rates[i])

    print(f"===Medidas de tendencia central, usando la media para el conjunto de pruebas {i+1}===")
    print(f"Media de tiempos de codificación: {mean_encoding:.2f}")
    print(f"Media de tiempos de decodificación: {mean_decoding:.2f}")
    print(f"Media de tasas de compresión: {mean_compression:.2f}")

    median_encoding = np.median(encoding_times[i])
    median_decoding = np.median(decoding_times[i])
    median_compression = np.median(compression_rates[i])

    print(f"===Medidas de tendencia central, usando la mediana para el conjunto de pruebas {i+1}===")
    print(f"Mediana de tiempos de codificación: {median_encoding:.2f}")
    print(f"Mediana de tiempos de decodificación: {median_decoding:.2f}")
    print(f"Mediana de tasas de compresión: {median_compression:.2f}")

    # Medidas de variabilidad
    std_dev_encoding = np.std(encoding_times[i])
    std_dev_decoding = np.std(decoding_times[i])
    std_dev_compression = np.std(compression_rates[i])

    print(f"===Medidas de variabilidad, usando la desviación estándar para el conjunto de pruebas {i+1}===")
    print(f"Desviación estándar de tiempos de codificación: {std_dev_encoding:.2f}")
    print(f"Desviación estándar de tiempos de decodificación: {std_dev_decoding:.2f}")
    print(f"Desviación estándar de tasas de compresión: {std_dev_compression:.2f}")

    # Pruebas de normalidad
    k2_encoding, p_encoding = stats.normaltest(encoding_times[i])
    k2_decoding, p_decoding = stats.normaltest(decoding_times[i])
    k2_compression, p_compression = stats.normaltest(compression_rates[i])

    print(f"===Pruebas de normalidad para el conjunto de pruebas {i+1}===")
    print(f"p-value de tiempos de codificación: {p_encoding:.2f}")
    print(f"p-value de tiempos de decodificación: {p_decoding:.2f}")
    print(f"p-value de tasas de compresión: {p_compression:.2f}")

# Gráficas
fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(221, projection='3d')

# Calcula el histograma 2D
hist, xedges, yedges = np.histogram2d(encoding_times.flatten(), decoding_times.flatten(), bins=20)

# Construye las coordenadas x, y
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construye las alturas dx, dy
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_xlabel('Encoding Times')
ax.set_ylabel('Decoding Times')
ax.set_zlabel('Compression Rates')
ax.set_title('Encoding, Decoding times and Compression rates distribution')

ax = fig.add_subplot(222, projection='3d')
ax.scatter(encoding_times.flatten(), decoding_times.flatten(), compression_rates.flatten())
ax.set_xlabel('Encoding Times')
ax.set_ylabel('Decoding Times')
ax.set_zlabel('Compression Rates')
ax.set_title('Encoding times vs Decoding times vs Compression rates')

plt.tight_layout()
plt.show()
