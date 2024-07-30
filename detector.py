#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image


# In[2]:


# Пути к вашим данным
train_path = 'C:/Users/alexs/Downloads/Training'
test_path = 'C:/Users/alexs/Downloads/Test'

size = (100, 100)  # Размер для изменения изображений
num_classes = 15   # Количество классов


# In[3]:


def load_data(data_path, size):
    classes = os.listdir(data_path)
    img_arr = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            img_arr.append(image)
            labels.append(class_idx)

    img_arr = torch.stack(img_arr)
    labels = torch.tensor(labels)
    return img_arr, labels


# In[4]:


x_train, y_train = load_data(train_path, size)
x_test, y_test = load_data(test_path, size)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# In[5]:


# Определение модели
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # Входные изображения имеют 3 канала (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.fc1 = nn.Linear(256 * 9 * 9, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, num_classes)  # Изменено на количество классов

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 256 * 9 * 9)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# In[6]:


# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Обучение на устройстве: {device}')


# In[22]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

epochs = 1000
best_accuracy = 0

for e in range(1, epochs + 1):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if e % 100 == 0:  # Изменено условие для вывода каждые 100 эпох
        print(f'\nEpoch {e}:')
        print('Evaluating:')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'Neirowebfruit.pth')

        scheduler.step()

print(f'\nBest accuracy: {best_accuracy}')

model.load_state_dict(torch.load('Neirowebfruit.pth'))


# In[7]:


# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Обучение на устройстве: {device}')

# Загрузка модели с корректным маппингом устройства
model = ConvNet().to(device)
model_path = 'Neirowebfruit.pth'

if device.type == 'cuda':
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# In[41]:


def classify_and_draw(image_path):
  
    pil_image = Image.open(image_path).convert('RGB')
    image = np.array(pil_image)  

    # Создание копии изображения 
    original_image = image.copy()

    # Преобразование изображения в тензор и классификация
    image_tensor = transform(pil_image).unsqueeze(0)  
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = classes[predicted.item()]

    # Заголовок с предсказанным классом
    cv2.putText(image, "Lemon", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  

  
    expected_region = [50, 50, 150, 150]
    # Выделение области изображения, в которой мы ожидаем объект
    x, y, w, h = expected_region
    region_of_interest = original_image[y:y+h, x:x+w]

    # Поиск контуров объектов в области интереса
    contours, _ = cv2.findContours(cv2.cvtColor(region_of_interest, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Находим самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Вычисляем ограничивающий прямоугольник для самого большого контура
    x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(largest_contour)
    
    # Добавляем смещение координат контура к общим координатам области интереса
    x_contour += x
    y_contour += y
    # Обводим самый большой контур
    cv2.rectangle(image, (x_contour, y_contour), (x_contour + w_contour, y_contour + h_contour), (0, 300, 0), 2)

    # Отображение изображения
    cv2.imshow('Classification Result',cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Использование
image_path = 'C:/Users/alexs/Downloads/photo_2023-03-21_12-30-07-e1679395845540.jpg'
classify_and_draw(image_path)

