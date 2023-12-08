import mlml as ml
import pickle
import cv2
import serial as sr
ser = sr.Serial('/dev/ttyUSB0', 9600,timeout= 1)
ser.flush()

while True:
    vid= cv2.VideoCapture(0)
    img,frames=vid.read()
    image1=frames
    cv2.imwrite('image.jpg', image1)
    dir = "D:/AI AND ROBOTICS/4th sem/minor project/PROCESS/Test_dataset/7486e823-64f7-4e43-ab51-26261b077fc2___RS_Early.B 6785.JPG"
    dir = "image.jpg"
    image = cv2.imread(dir, 0)
    leafe_dis = cv2.resize(image, (50, 50))

    image = leafe_dis.reshape(1, -1)

    pick = open("model1.sav", "rb")
    model = pickle.load(pick)
    pick.close()

    predicted = []
    accu = []

    prediction = model.predict(image)

    categories = ["corn_Blight", "corn_common_Rust", 'corn_gray_Leaf_Spot', 'corn_healthy', 'Potato_Early_blight',
                  'Potato_healthy',
                  'Potato_Late_blight', 'rice_bacterial_leaf_blight', 'rice_brown_spot', "rice_healthy",
                  'rice_leaf_smut', 'Tomato_Bacterial_spot',
                  'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                  'Tomato_Septoria_leaf_spot',
                  'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_mosaic_virus',
                  'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'wheat_septoria', 'wheat_stripe_rust', 'wheat_Healthy']

    # perform = categories[prediction[0]]
    # print(perform)

    # predicted.append(perform)
    #
    # output = image.reshape(50,50)
    # plt.imshow(output , cmap="gray")
    # plt.show()
    #
    #
    # print(predicted, end=" ")
    # print("\n",accu)
    # my = "Category: " + perform + " \n" + ml.disease[perform]

    # if perform in ml.disease:
    #     ml.msg(my)
    line = ser.readline().decode('utf-8').rsplit()
    if line == 1:
        perform = categories[prediction[0]]
        my = "Category: " + perform + " \n" + ml.disease[perform]
        print(perform)
        if perform in ml.disease:
            ml.msg(my)
    else:
        pass
