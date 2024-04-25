import matplotlib.pyplot as plt

x = [1,2,3,4,5]
h = [10,20,30,40,50]
color = ['r','b','g','y','m']   # 顏色數據
label = ['a','b','c','d','e']   # 標籤數據
plt.bar(x,h,color=color,tick_label=label,width=0.5)  # 加入顏色、標籤和寬度參數
plt.show()

# a = []

# print(a)
# print(type(a))