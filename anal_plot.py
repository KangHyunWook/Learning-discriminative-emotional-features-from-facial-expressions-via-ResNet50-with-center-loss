import matplotlib.pyplot as plt

f = open('/home/jeff/projects/Journal of KIISE/KCI2023-main/results/cent-weight-0.2/RAF-DB_results.csv', 'r')

raf_db_alphas=[]
raf_db_accs=[]

first_line_flag=0
for line in f:
    line=line.strip()
    if first_line_flag==0:
        first_line_flag=1
        continue
    splits = line.split(',')
    raf_db_alphas.append(float(splits[0]))
    raf_db_accs.append(float(splits[2]))

f= open('/home/jeff/projects/Journal of KIISE/KCI2023-main/results/cent-weight-0.2/FER2013-results.csv', 'r')
fer_alphas=[]
fer_accs=[]
first_line_flag=0
for line in f:
    line=line.strip()
    if first_line_flag==0:
        first_line_flag=1
        continue
    splits = line.split(',')
    fer_alphas.append(float(splits[0]))
    fer_accs.append(float(splits[2]))



# plt.set_title('Respective performances according to different balancing factors for the center loss.')


plt.plot(fer_alphas, fer_accs, label='FER2013')
plt.plot(raf_db_alphas, raf_db_accs, label='RAF-DB')
plt.ylabel('classification accuracies (%)')
plt.xlabel('alpha value')
plt.legend(bbox_to_anchor=(1.3, 1.0), loc='upper right')
plt.tight_layout()
plt.show()









#
