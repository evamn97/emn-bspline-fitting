from utils import *

if __name__ == '__main__':

    filename = "example_profile.txt"
    profile = pd.read_csv(filename, delimiter=',', names=['x', 'z'])

    plot_data_only(profile, title="Example Input Profile", save_to='')
