from scipy.io import loadmat

def load_clean_data():
    data = loadmat('data/cleandata_students.mat')
    return data['x'], data['y']

def load_noisy_data():
    data = loadmat('data/noisydata_students.mat')
    return data['x'], data['y']

if __name__ == '__main__':
    print('Clean Data:')
    x, y = load_clean_data()
    print('x:', x)
    print('y:', y)

    print()

    print('Noisy Data:')
    x, y = load_noisy_data()
    print('x:', x)
    print('y:', y)
