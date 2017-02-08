from scipy.io import loadmat


def load_data(data_file):
    data = loadmat(data_file)
    return data['x'], data['y']


if __name__ == '__main__':
    print('Clean Data:')
    x, y = load_data('data/cleandata_students.mat')
    print('x:', x)
    print('y:', y)

    print()

    print('Noisy Data:')
    x, y = load_data('data/noisydata_students.mat')
    print('x:', x)
    print('y:', y)
