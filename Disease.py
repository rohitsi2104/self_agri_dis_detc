import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random


# path = os.path.join(dir)
# leafe = cv2.imread(path , 0)
# lola = cv2.resize(leafe, (256,256))
# image = lola.reshape(1,-1)

# dir = "D:\\AI AND ROBOTICS\\4th sem\\minor project\\Leaf_dataset"
#
# categories = ["corn_Blight","corn_common_Rust",'corn_gray_Leaf_Spot','corn_healthy','Potato_Early_blight','Potato_healthy',
#               'Potato_Late_blight','rice_bacterial_leaf_blight','rice_brown_spot',"rice_healthy",'rice_leaf_smut','Tomato_Bacterial_spot',
#               'Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
#               'Tomato_Spider_mites Two-spotted_spider_mite','Tomato_Target_Spot','Tomato_Tomato_mosaic_virus',
#               'Tomato_Tomato_Yellow_Leaf_Curl_Virus','wheat_septoria','wheat_stripe_rust','wheat_Healthy']
# #
# data = []
# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)
#
#     for img in os.listdir(path):
#         imgpath = os.path.join(path, img)
#         leaf_image =cv2.imread(imgpath,0)
#         try:
#             leaf_image = cv2.resize(leaf_image,(50,50))
#             image = np.array(leaf_image).flatten()
#             data.append([image, label])
#         except Exception as e:
#             pass


# pick_in = open("data_leaf.pickle", "wb")
# pickle.dump(data, pick_in)
# pick_in.close()

pick = open("data_leaf.pickle", "rb")
data = pickle.load(pick)
pick.close()


#
random.shuffle(data)
feature = []
label = []
#
for features , labels in data:
    feature.append(features)
    label.append(labels)

x_train , x_test , y_train , y_test = train_test_split(feature , label, test_size= 0.25)

# model = SVC(C=1, kernel="poly", gamma="auto")
# model.fit(x_train, y_train)


# pick = open("model1.sav", "wb")
# pickle.dump(model, pick)
# pick.close()

pick = open("model1.sav", "rb")
model = pickle.load(pick)
pick.close()

predicted = []
accu = []

prediction = model.predict(x_test)
accuracy = model.score(x_test,y_test)

categories = ["corn_Blight","corn_common_Rust",'corn_gray_Leaf_Spot','corn_healthy','Potato_Early_blight','Potato_healthy',
              'Potato_Late_blight','rice_bacterial_leaf_blight','rice_brown_spot',"rice_healthy",'rice_leaf_smut','Tomato_Bacterial_spot',
              'Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
              'Tomato_Spider_mites Two-spotted_spider_mite','Tomato_Target_Spot','Tomato_Tomato_mosaic_virus',
              'Tomato_Tomato_Yellow_Leaf_Curl_Virus','wheat_septoria','wheat_stripe_rust','wheat_Healthy']

perform = categories[prediction[0]]

print("Accuracy : ", accuracy)
print("prediction  is : ",perform)

predicted.append(perform)
accu.append(accuracy)

output = x_test[0].reshape(50,50)
plt.imshow(output , cmap="gray")
plt.show()


print(predicted, end=" ")
print("\n",accu)