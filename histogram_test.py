import matplotlib.pyplot as plt

x = [1,2,3,4,5]
h = [10,20,30,40,50]
color = ['r','b','g','y','m']
label = ['affff','beaef','cfewfa','dfewafewf','efewfa']
# plt.figure(figsize=(6,6))
plt.bar(x,h,color=color,tick_label=label,width=0.5)
plt.tick_params(axis='x', labelrotation=70)
plt.xlabel('gykugky')
plt.tight_layout()
plt.show()

# a = []

# print(a)
# print(type(a))