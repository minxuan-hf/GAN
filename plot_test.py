import matplotlib.pyplot as plt

def plot_show(input_list, title, name):
    plt.plot(input_list)
    plt.title(title)
    plt.savefig(name,dpi=600)
    plt.show()
