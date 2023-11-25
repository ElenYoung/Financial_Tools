import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# 创建3D坐标轴

def plt_3d_bar(result,index_name,col_name,key_name,title = False,alpha=0.8):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # x, y, z轴的数据
    x_labels = result.index
    y_labels = result.columns
    x_pos = np.arange(len(x_labels))
    y_pos = np.arange(len(y_labels))
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    
    # 随机生成一些数据来表示z轴的高度
    z_pos = np.zeros(x_pos.shape)
    dx = dy = 0.6
    dz = result.to_numpy()
    
    #color
    color = ['b','y','r','c','m','orange','g']
    
    # 绘制柱状图
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.bar3d(x_pos[j, i], y_pos[j, i], z_pos[j, i], dx, dy, dz[j, i],color=color[j],alpha=alpha)
    
    # 设置x轴和y轴的标签
    ax.set_xticks(x_pos.flatten()[0:len(x_labels)] + dx/2)
    ax.set_yticks(y_pos.T.flatten()[0:len(y_labels)] + dy/2)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # 设置标签和标题
    ax.set_xlabel(index_name)
    ax.set_ylabel(col_name)
    ax.set_zlabel(key_name)
    if title != False:
        ax.set_title(title)

    
    plt.show()
    
    
    
    
