from funcs_transmorph import *
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    BrightestCenterSquareCrop(),
    transforms.Resize((64, 64)),
])


train_dataset = imagepairdataset(root_dir='train', transform = transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

print(len(next(iter(train_loader))))
print(next(iter(train_loader))[0].shape)

imgs = next(iter(train_loader))
a = imgs[0].numpy()
b = imgs[1].numpy()

plt.subplot(1,2,1)
plt.imshow(np.squeeze(a))

plt.subplot(1,2,2)
plt.imshow(np.squeeze(b))
plt.show()
