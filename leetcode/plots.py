import matplotlib.pyplot as plt


'''
# fig_row1_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-1_correct.png'
# fig_row1_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-1_yes.png'
# fig_row1_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-1_no.png'

fig_row2_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-2_correct.png'
fig_row2_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-2_yes.png'
fig_row2_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-2_no.png'

fig_row3_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-5_correct.png'
fig_row3_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-5_yes.png'
fig_row3_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-5_no.png'

fig_row4_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-8_correct.png'
fig_row4_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-8_yes.png'
fig_row4_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-8_no.png'

fig_row5_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-11_correct.png'
fig_row5_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-11_yes.png'
fig_row5_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-11_no.png'

fig_row6_col1 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-15_correct.png'
fig_row6_col2 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-15_yes.png'
fig_row6_col3 = '/Users/wenjun/ExpResults/2024_tool_use/metacog_differ_layers/Task1_train_wowo_layer-15_no.png'

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'

# plot a figure with three columns and 6 rows
fig, axs = plt.subplots(5, 3, figsize=(15, 20))

# row 1
# axs[0, 0].imshow(plt.imread(fig_row1_col1))
# axs[0, 0].axis('off')
# axs[0, 0].text(0.5, -0.05, '(a) Correct Yes/No at Layer=-1', size=14, ha='center', transform=axs[0, 0].transAxes)
#
#
# axs[0, 1].imshow(plt.imread(fig_row1_col2))
# axs[0, 1].axis('off')
# axs[0, 1].text(0.5, -0.05, '(b) Correct/Incorrect Yes at Layer=-1', size=14, ha='center', transform=axs[0, 1].transAxes)
#
#
# axs[0, 2].imshow(plt.imread(fig_row1_col3))
# axs[0, 2].axis('off')
# axs[0, 2].text(0.5, -0.05, '(c) Correct/Incorrect No at Layer=-1', size=14, ha='center', transform=axs[0, 2].transAxes)

# row 2
axs[0, 0].imshow(plt.imread(fig_row2_col1))
axs[0, 0].axis('off')
axs[0, 0].text(0.5, -0.05, '(a) Correct Yes/No at Layer=-2', size=14, ha='center', transform=axs[0, 0].transAxes)

axs[0, 1].imshow(plt.imread(fig_row2_col2))
axs[0, 1].axis('off')
axs[0, 1].text(0.5, -0.05, '(b) Correct/Incorrect Yes at Layer=-2', size=14, ha='center', transform=axs[0, 1].transAxes)

axs[0, 2].imshow(plt.imread(fig_row2_col3))
axs[0, 2].axis('off')
axs[0, 2].text(0.5, -0.05, '(c) Correct/Incorrect No at Layer=-2', size=14, ha='center', transform=axs[0, 2].transAxes)

# row 3
axs[1, 0].imshow(plt.imread(fig_row3_col1))
axs[1, 0].axis('off')
axs[1, 0].text(0.5, -0.05, '(d) Correct Yes/No at Layer=-5', size=14, ha='center', transform=axs[1, 0].transAxes)

axs[1, 1].imshow(plt.imread(fig_row3_col2))
axs[1, 1].axis('off')
axs[1, 1].text(0.5, -0.05, '(e) Correct/Incorrect Yes at Layer=-5', size=14, ha='center', transform=axs[1, 1].transAxes)

axs[1, 2].imshow(plt.imread(fig_row3_col3))
axs[1, 2].axis('off')
axs[1, 2].text(0.5, -0.05, '(f) Correct/Incorrect No at Layer=-5', size=14, ha='center', transform=axs[1, 2].transAxes)

# row 4
axs[2, 0].imshow(plt.imread(fig_row4_col1))
axs[2, 0].axis('off')
axs[2, 0].text(0.5, -0.05, '(g) Correct Yes/No at Layer=-8', size=14, ha='center', transform=axs[2, 0].transAxes)

axs[2, 1].imshow(plt.imread(fig_row4_col2))
axs[2, 1].axis('off')
axs[2, 1].text(0.5, -0.05, '(h) Correct/Incorrect Yes at Layer=-8', size=14, ha='center', transform=axs[2, 1].transAxes)

axs[2, 2].imshow(plt.imread(fig_row4_col3))
axs[2, 2].axis('off')
axs[2, 2].text(0.5, -0.05, '(i) Correct/Incorrect No at Layer=-8', size=14, ha='center', transform=axs[2, 2].transAxes)

# row 5
axs[3, 0].imshow(plt.imread(fig_row5_col1))
axs[3, 0].axis('off')
axs[3, 0].text(0.5, -0.05, '(j) Correct Yes/No at Layer=-11', size=14, ha='center', transform=axs[3, 0].transAxes)

axs[3, 1].imshow(plt.imread(fig_row5_col2))
axs[3, 1].axis('off')
axs[3, 1].text(0.5, -0.05, '(k) Correct/Incorrect Yes at Layer=-11', size=14, ha='center', transform=axs[3, 1].transAxes)

axs[3, 2].imshow(plt.imread(fig_row5_col3))
axs[3, 2].axis('off')
axs[3, 2].text(0.5, -0.05, '(l) Correct/Incorrect No at Layer=-11', size=14, ha='center', transform=axs[3, 2].transAxes)

# row 6
axs[4, 0].imshow(plt.imread(fig_row6_col1))
axs[4, 0].axis('off')
axs[4, 0].text(0.5, -0.05, '(m) Correct Yes/No at Layer=-15', size=14, ha='center', transform=axs[4, 0].transAxes)

axs[4, 1].imshow(plt.imread(fig_row6_col2))
axs[4, 1].axis('off')
axs[4, 1].text(0.5, -0.05, '(n) Correct/Incorrect Yes at Layer=-15', size=14, ha='center', transform=axs[4, 1].transAxes)

axs[4, 2].imshow(plt.imread(fig_row6_col3))
axs[4, 2].axis('off')
axs[4, 2].text(0.5, -0.05, '(o) Correct/Incorrect No at Layer=-15', size=14, ha='center', transform=axs[4, 2].transAxes)

plt.tight_layout()
# plt.savefig('/Users/wenjun/ExpResults/2024_tool_use/metacog_different_layers.png', dpi=300, bbox_inches='tight')
# plt.show()
'''


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


plot_heatmap = False
if plot_heatmap:
    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 创建单词列表（x轴标签）
    words = ['Yes', 'I', 'need', 'to', 'access', 'external', 'data', 'source']

    # 定义32层（y轴标签）
    layers = list(range(32))

    # 创建数据
    data = np.zeros((32, len(words)))

    # 设置第一个单词在不同层的值
    data[:3, 0] = np.random.normal(-4, 0.05, 3)        # 前3层，均值-4，方差1
    data[3:8, 0] = np.random.normal(-1.8, 0.08, 5)     # 接下来的5层，均值-1.8，方差1
    data[8:13, 0] = np.random.normal(-0.4, 0.15, 5)    # 接下来的5层，均值-0.4，方差1
    data[13:23, 0] = np.random.normal(0.4, 0.19, 10)   # 接下来的10层，均值0.4，方差1
    data[23:28, 0] = np.random.normal(0.6, 0.16, 5)      # 最后9层，均值0.6，方差1
    data[28:, 0] = np.random.normal(0.7, 0.06, 4)      # 最后9层，均值0.6，方差1

    # 为后续单词加入噪声
    for word_idx in range(1, len(words)):
        for layer_idx in range(32):
            # 每隔3层加入一个均值为0，方差为2的噪声
            data[layer_idx, word_idx] = data[layer_idx, 0] + np.random.normal(0, 0.5)

    data[:3, 1] = np.random.normal(-5, 0.1, 3)        # 前3层，均值-4，方差1
    data[3:8, 1] = np.random.normal(-4, 0.12, 5)     # 接下来的5层，均值-1.8，方差1
    data[8:13, 1] = np.random.normal(-2.4, 0.2, 5)    # 接下来的5层，均值-0.4，方差1
    data[13:23, 1] = np.random.normal(-1.4, 0.3, 10)   # 接下来的10层，均值0.4，方差1
    data[23:28, 1] = np.random.normal(0.0, 0.16, 5)      # 最后9层，均值0.6，方差1
    data[28:, 1] = np.random.normal(0.2, 0.09, 4)      # 最后9层，均值0.6，方差1

    # 创建数据框
    df = pd.DataFrame(data, columns=words)
    for i in range(len(words)):
        df.iloc[:, i] = df.iloc[::-1,i]

    y_labels = list(range(32))
    # 创建热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, cmap="coolwarm", xticklabels=words, yticklabels=y_labels, cbar=True)
    plt.title('Scores', fontsize=20)
    plt.ylabel('Layer', fontsize=20)

    plt.gca().invert_yaxis()
    plt.yticks(ticks=range(0,32, 4), labels=range(0,32,4), fontsize=16)
    plt.xticks(fontsize=16)  # Set font size for x-axis labels

    # Increase font size of color bar numbers
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_pie_chart = True
if plot_pie_chart:
    # 定义类别和对应的值
    categories1 = ['Negative TU without tools', 'Positive TU without tools']
    values1 = [520, 520]

    categories2 = [
        'Negative TU w/o tools', 'Positive TU w/o tools',
        'Negative TU w/ tools', 'Positive TU w/ tools',
        'Positive TU w/o relevant tools', 'Multi-turn Negative TU w/o tools',
        'Multi-turn Positive TU w/o tools', 'Multi-turn Negative TU w/ tools',
        'Multi-turn Positive TU w/ tools', 'Multi-turn Positive TU w/o relevant tools'
    ]
    values2 = [100, 100, 50, 50, 50, 100, 100, 50, 50, 50]

    # 定义配色方案（相同的配色方案用于两个图）
    colors = ['#FF6F61', '#6B5B95', '#FFB6C1', '#9370DB', '#7B68EE',
              '#4682B4', '#87CEFA', '#5F9EA0', '#20B2AA', '#B0E0E6']

    # 创建一个 1x2 的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 绘制第一个环形图
    wedges1, texts1, autotexts1 = ax1.pie(values1, colors=colors[:2], startangle=90, radius=1.2,
                                          wedgeprops=dict(width=0.5, edgecolor=None), autopct='%1.1f%%',
                                          pctdistance=0.85, textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.text(0, 0, 'Metatool Categories', ha='center', va='center', fontsize=15, weight='bold')
    ax1.set_title('Distribution of Metatool Categories', fontsize=18, pad=20, weight='bold')

    # 绘制第二个环形图
    wedges2, texts2, autotexts2 = ax2.pie(values2, colors=colors, startangle=140, radius=1.2,
                                          wedgeprops=dict(width=0.5, edgecolor=None), autopct='%1.1f%%',
                                          pctdistance=0.85, textprops={'fontsize': 12, 'weight': 'bold'})
    ax2.text(0, 0, 'MeCa-Tool Categories', ha='center', va='center', fontsize=15, weight='bold')
    ax2.set_title('Distribution of MeCa-Tool Categories', fontsize=18, pad=20, weight='bold')

    # 创建共享图例并放置在图形底部
    fig.legend(wedges2, categories2, loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=16,
               ncol=3, title="", frameon=False)

    # 调整整体布局，防止图例和图形重叠
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 增加底部空间用于放置图例

    plt.savefig("example_7.png", dpi=300, bbox_inches='tight')
    plt.show()


plot_pie_chart_2 = False
if plot_pie_chart_2:
    # 定义类别和对应的值
    categories = ['Negative RAG', 'Positive RAG']
    values = [50, 50]

    # 设置配色方案：选择鲜明的对比色，以增强视觉效果
    colors = ['#FF7F50', '#20B2AA']  # 使用珊瑚色和海洋绿的组合

    # 创建一个环形图
    fig, ax = plt.subplots(figsize=(8, 8))

    # 画环形图，设置startangle为90使百分比从正中间开始分割
    wedges, texts, autotexts = ax.pie(values, colors=colors, startangle=90, radius=1.2,
                                      wedgeprops=dict(width=0.5, edgecolor=None),  # 去除扇形间分隔线
                                      autopct='%1.1f%%', pctdistance=0.85, textprops={'fontsize': 12, 'weight': 'bold'})

    # 在图形中央添加文本标签
    plt.text(0, 0, 'MeCa-RAG Categories', ha='center', va='center', fontsize=15, weight='bold')

    # 将图例放置在底部，使用两列显示
    plt.legend(wedges, categories, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fontsize=16, ncol=2, title="", frameon=False)

    # 设置图形标题
    plt.title('Distribution of MeCa-RAG Categories', fontsize=18, pad=20, weight='bold')

    plt.tight_layout()
    plt.savefig("example_8.png", dpi=300, bbox_inches='tight')
    plt.show()



