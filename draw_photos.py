# 读取txt文档，并利用plt库绘制图像
import matplotlib.pyplot as plt


def plot_show(input_list, title, name):
    plt.plot(input_list, color='#023e8a')
    plt.xlabel('Epochs',
               fontdict={'fontsize': 15, 'fontstyle': 'oblique', 'color': 'red'})
    plt.ylabel('Loss',
               fontdict={'fontsize': 15, 'fontstyle': 'oblique', 'color': 'red'})
    plt.title(title, fontsize='xx-large', fontweight='heavy', color='black')
    plt.savefig(name, dpi=600)
    plt.show()


# 读取txt文档，并存储到列表中
def read_txt(file_path):
    f = open(file_path)
    txt = []
    for line in f:
        txt.append(float(line.strip()))
    return txt

if __name__=="__main__":
    plot_show(read_txt('220607_ROP_txt\g.txt'), "Generator_Loss", "220607_ROP_resultspic\Generator_Loss.png")
