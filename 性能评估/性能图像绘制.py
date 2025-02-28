import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
data = pd.read_csv('./BER_comparison.csv')

# 绘制三条曲线
plt.plot(data["Epsilon"], data["LBC_BER"], label='Linear Block Code', marker='o')
plt.plot(data["Epsilon"], data["CC_BER"], label='Convolutional Code', marker='s')
plt.plot(data["Epsilon"], data["NoCode_BER"], label='No Encoding', marker='x')

# 图形美化
plt.xlabel('Epsilon (Error Probability)')
plt.ylabel('BER (Bit Error Rate)')
plt.title('BER vs Epsilon: Comparison of Different Encoding Schemes')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
