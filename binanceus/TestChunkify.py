
import numpy as np


def chunkify(data, seq_len):
    # input format = [nrows, nfeatures] output = [nrows, seq_len, nfeatures]
    nrows = np.shape(data)[0]
    nfeatures = np.shape(data)[1]
    chunked_array = np.zeros((nrows, seq_len, nfeatures), dtype=float)
    zero_row = np.zeros((nfeatures), dtype=float)
    # chunked_array = []

    reverse = True

    # fill the first part (0..seqlen rows), which are only sparsely populated
    for row in range(seq_len):
        for seq in range(seq_len):
                if seq >= (seq_len-row-1):
                    chunked_array[row][seq] = data[(row+seq)-seq_len+1]
                else:
                    chunked_array[row][seq] = zero_row
        if reverse:
            chunked_array[row] = np.flipud(chunked_array[row])


    # fill the rest
    # print("Data:{}, len:{}".format(np.shape(data), seq_len))
    for row in range(seq_len, nrows):
        chunked_array[row] = data[(row - seq_len) + 1:row + 1]
        if reverse:
            chunked_array[row] = np.flipud(chunked_array[row])


    # print("data: ", data)
    # print("chunked: ", chunked_array)
    print("data:{} chunked:{}".format(np.shape(data), np.shape(chunked_array)))
    return chunked_array

def main():

    nrows = 12
    nfeatures = 6

    test_array = np.zeros((nrows, nfeatures), dtype=float)

    for i in range(nrows):
        for j in range(nfeatures):
            test_array[i][j] = float(i+1)

    chunky = chunkify(test_array, 4)

    print("test_array:")
    print(test_array)
    print("chunky:")
    for i in range(nrows):
        print("[", i, "]")
        print(chunky[i])

if __name__ == '__main__':
    main()