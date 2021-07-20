import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import ast, os


st.title("Indian Currency Classifier")
st.header("Classifies types of Indian Currency 10,20,50,100,200,500.2000")
st.text("Provide image and path of Indian Currency note")

model = load_model('currency_mobilenetmodel.h5')
f = open("mobilenet_currency_class_indices.txt", "r")
labels = f.read()
labels = ast.literal_eval(labels)
final_labels = {v: k for k, v in labels.items()}

model_watermark = load_model('watermark_mobilenetmodel.h5')
f_w = open("mobilenet_watermark_class_indices.txt", "r")
labels_w = f_w.read()
labels_w = ast.literal_eval(labels_w)
final_labels_w = {v: k for k, v in labels_w.items()}

from img_classification import currency_classification

def predict_image(imgname, from_test_dir):
    test_image = image.load_img(imgname, target_size = (224, 224))

    # plt.imshow(test_image)
    # plt.show()

    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = (2.0 / 255.0) * test_image - 1.0
    result = model.predict(test_image)

    result_dict = dict()
    for key in list(final_labels.keys()):
        result_dict[final_labels[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    if not from_test_dir:
        print('=' * 50)
        for label in sorted_results.keys():
            print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result
    

def predict_image_watermark(imgname, from_test_dir):
    test_image = image.load_img(imgname, target_size = (224, 224))

    # plt.imshow(test_image)
    # plt.show()

    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = (2.0 / 255.0) * test_image - 1.0
    result = model_watermark.predict(test_image)

    result_dict = dict()
    for key in list(final_labels_w.keys()):
        result_dict[final_labels_w[key]] = result[0][key]
    sorted_results = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    if not from_test_dir:
        print('=' * 50)
        for label in sorted_results.keys():
            print("{}: {}%".format(label, sorted_results[label] * 100))

    final_result = dict()
    final_result[list(sorted_results.keys())[0]] = sorted_results[list(sorted_results.keys())[0]] * 100

    return final_result

file_path = st.text_input("enter file path for classification")
file_path_w = st.text_input("enter file path for validation")
uploaded_file = st.file_uploader("upload here",type="jpg")
if uploaded_file is not None:
    image_test = Image.open(uploaded_file)
    st.image(image_test, caption='Uploaded Note', use_column_width=True)
    st.write("")
    st.write("test file path for classification", file_path)
    st.write("test file path for validation", file_path_w)
    st.write("Classifying and Validating...")
    st.write("Just a minute")

    final_result_classification = predict_image(file_path, False)
    final_result_validation = predict_image_watermark(file_path_w, False)
    # label = currency_classification(image, 'currency_mobilenetmodel.h5')
    # switcher = {
    #          0 : "fifty", 
    #          1: "fivehundred",
    #          2: "hundred",
    #          3: "ten",
    #          4:"thousand",
    #          5:"twenty"
    # }
    # s=switcher.get(label, "Not Maching")
    # st.write("Done..")
    # if s=="Not Maching":
    #     st.write("Enter valid Image")
    # else :
    #     st.write("This is ", s," rupees note")

    st.write("The image uploaded is \n", list(final_result_classification.keys())[0])
    st.write("The image validation is \n", list(final_result_validation.keys())[0])


