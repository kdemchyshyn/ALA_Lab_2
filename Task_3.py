import numpy as np

def encrypt_message(message, key_matrix):

    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)

    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):

    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    inverse_diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(1/eigenvalues)), np.linalg.inv(eigenvectors))
    message_vector = np.dot(inverse_diagonalized_key_matrix, encrypted_vector)
    message_vector = np.round(message_vector).astype(int)
    message = ''.join(chr(x) for x in message_vector)

    return message

def task3(message):
    key_matrix = np.random.randint(0, 256, (len(message), len(message)))

    encrypted_message = encrypt_message(message, key_matrix)
    print(encrypted_message)

    decrypted_message = decrypt_message(encrypted_message, key_matrix)
    print(decrypted_message)
