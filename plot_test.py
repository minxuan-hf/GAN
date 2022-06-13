"""
_*_coding: utf-8_*_
Editor: minxuan
Date: 2022.01.20
"""
import matplotlib.pyplot as plt

# m = [2, 3, -1, 1, -2]
# plt.plot(m)
# plt.savefig("plot_test.png")
# plt.show()


def plot_show(input_list, title, name):
    plt.plot(input_list)
    plt.title(title)
    plt.savefig(name,dpi=600)
    plt.show()

# plot_show([12,23,23,11,22],"nums","plt_test2.png")