import numpy as np
import string

# Encoding the words

def string_encoding(word_dict: np.ndarray) -> np.ndarray:
    char_to_idx = {char : i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inps = []
    for row in word_dict:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_idx:
                one_hot_vec = np.zeros((len(string.ascii_uppercase),1))
                one_hot_vec[char_to_idx[char.upper()]] = 1
                one_hot_list.append(one_hot_vec)
        one_hot_inps.append(one_hot_list)


    return np.array(one_hot_inps)


inputs = np.array([
    ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
    ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
    ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
    ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
])

