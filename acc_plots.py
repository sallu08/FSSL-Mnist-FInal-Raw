
import matplotlib.pyplot as plt
import csv

loss = []
acc = []
val_loss = []
val_acc = []

with open('historya.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for i, row in enumerate(plots):
        if i%2!=0:
            if row[1]!="loss":
                print("loss", row[1])
                loss.append(float(row[1]))
                
            if row[2]!="acc":
                print("acc", row[2])
                acc.append(float(row[2]))
                
            if row[3]!="val_loss":
                print("val_loss", row[3])
                val_loss.append(float(row[3]))
                
            if row[4]!="val_acc":
                print("val_acc", row[4])
                val_acc.append (float(row[4]))

            # print(row)
            # print(np.array(row[0]).shape)
fig, ax = plt.subplots(2,1)
ax[0].plot(loss, color='r', label="Training Loss")
ax[0].plot(val_loss, color='b', label="Validation Loss",axes =ax[0])
ax[1].plot(acc, color='r', label="Training Acc")
ax[1].plot(val_acc, color='b', label="Validation Acc",axes =ax[1])

legend = ax[0].legend(loc='best', shadow=True)
legend = ax[1].legend(loc='best', shadow=True)
# plt.title('Hospital B')
plt.show(0)

# ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
# ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
# legend = ax[1].legend(loc='best', shadow=True)