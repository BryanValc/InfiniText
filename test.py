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

# We initialize the list of test texts
test_texts = []

# Generate samples of 3 to 6 words
for num_words in range(3, 7):
    for _ in range(100):
        start_index = random.randint(0, len(words) - num_words)
        sample = words[start_index:start_index + num_words]
        test_texts.append(' '.join(sample))
        # print(test_texts[-1])

# Load the model
model = main.load_model('model.json')

# Initialize the lists of times for encoding and decoding
encoding_times = []
decoding_times = []
compression_rates = []

# For each test text we encode and decode it, and measure the times
success_count = 0
failure_count = 0

for test_text in test_texts:
    original_size = len(test_text) * 8
    compression_rate = original_size / 98
    compression_rates.append(compression_rate)

    # encoding
    start_time = time.time()
    bit_string_letters, bit_string_length, bit_word_count = main.encode_target_text(test_text)
    submodel = main.create_submodel(model, bit_string_letters, bit_string_length)
    bit_word_index = main.get_word_index(test_text, submodel)
    seed = main.find_matching_seed(test_text, submodel, int(bit_word_index, 2), int(bit_word_count, 2))
    encoding_times.append(time.time() - start_time)

    # Decodificación
    start_time = time.time()
    decoded_text = main.recreate_text(model, bit_string_letters, bit_string_length, bit_word_count, bit_word_index, seed)
    decoding_times.append(time.time() - start_time)

    # Prueba de aserción
    if test_text != decoded_text:
        print(f"Error: {test_text} != {decoded_text}")
        failure_count += 1
    else:
        print(f"Success: {test_text} == {decoded_text}")
        success_count += 1
    # only 2 decimals for readability
    print(f"Encoding(ms): {encoding_times[-1] * 1000:.2f}")
    print(f"Decoding(ms): {decoding_times[-1] * 1000:.2f}")
    print(f"Compression rate: {compression_rate:.2f}")

print(f"Success count: {success_count}")
print(f"Failure count: {failure_count}")
print(f"Average encoding time: {sum(encoding_times) / len(encoding_times) * 1000:.2f} ms")
print(f"Average decoding time: {sum(decoding_times) / len(decoding_times) * 1000:.2f} ms")
print(f"Average compression rate: {sum(compression_rates) / len(compression_rates):.2f}")

# Convertir las listas a arrays de numpy para facilitar los cálculos
encoding_times = np.array(encoding_times)
decoding_times = np.array(decoding_times)
compression_rates = np.array(compression_rates)

# Medidas de tendencia central
mean_encoding = np.mean(encoding_times)
mean_decoding = np.mean(decoding_times)
mean_compression = np.mean(compression_rates)

print("===Medidas de tendencia central, usando la media===")
print(f"Media de tiempos de codificación: {mean_encoding:.2f}")
print(f"Media de tiempos de decodificación: {mean_decoding:.2f}")
print(f"Media de tasas de compresión: {mean_compression:.2f}")

median_encoding = np.median(encoding_times)
median_decoding = np.median(decoding_times)
median_compression = np.median(compression_rates)

print("===Medidas de tendencia central, usando la mediana===")
print(f"Mediana de tiempos de codificación: {median_encoding:.2f}")
print(f"Mediana de tiempos de decodificación: {median_decoding:.2f}")
print(f"Mediana de tasas de compresión: {median_compression:.2f}")

# Medidas de variabilidad
std_dev_encoding = np.std(encoding_times)
std_dev_decoding = np.std(decoding_times)
std_dev_compression = np.std(compression_rates)

print("===Medidas de variabilidad, usando la desviación estándar===")
print(f"Desviación estándar de tiempos de codificación: {std_dev_encoding:.2f}")
print(f"Desviación estándar de tiempos de decodificación: {std_dev_decoding:.2f}")
print(f"Desviación estándar de tasas de compresión: {std_dev_compression:.2f}")

# Pruebas de normalidad
k2_encoding, p_encoding = stats.normaltest(encoding_times)
k2_decoding, p_decoding = stats.normaltest(decoding_times)
k2_compression, p_compression = stats.normaltest(compression_rates)

print("===Pruebas de normalidad===")
print(f"p-value de tiempos de codificación: {p_encoding:.2f}")
print(f"p-value de tiempos de decodificación: {p_decoding:.2f}")
print(f"p-value de tasas de compresión: {p_compression:.2f}")

# Pruebas estadísticas
# Por ejemplo, puedes realizar una prueba t para comparar los tiempos de codificación y decodificación
t_statistic, p_value = stats.ttest_ind(encoding_times, decoding_times)

# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(encoding_times, bins=20, color='blue', alpha=0.7)
plt.title('Encoding times distribution')

plt.subplot(2, 2, 2)
plt.hist(decoding_times, bins=20, color='green', alpha=0.7)
plt.title('Decoding times distribution')

plt.subplot(2, 2, 3)
plt.scatter(encoding_times, decoding_times)
plt.xlabel('Encoding times')
plt.ylabel('Decoding times')
plt.title('Encoding times vs Decoding times')

plt.subplot(2, 2, 4)
plt.scatter(compression_rates, encoding_times)
plt.xlabel('Compression rates')
plt.ylabel('Encoding times')
plt.title('Compression rates vs Encoding times')

plt.tight_layout()
plt.show()
