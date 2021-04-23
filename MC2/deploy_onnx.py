import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import psutil
#import gpizero
import telnetlib as tel
import time
import threading
import json


total_power = 0
max_power = -1
max_mem = -1
DELAY = 0.02
def getTelnetPower(SP2_tel, last_power):    #for both devices
    tel_dat = str(SP2_tel.read_very_eager())
    #print("telnet reading:", tel_dat)
    findex = tel_dat.rfind('\n')
    findex2 = tel_dat[:findex].rfind('\n')
    findex2 = findex2 if findex2 != -1 else 0
    ln = tel_dat[findex2: findex].strip().split(',')
    if len(ln) < 2:
        power = last_power
    else:
        power = float(ln[-2])
    return power

def power_thread(SP2_tel, stop):
    global total_power
    global max_power
    global max_mem
    power = 0
    while True:
        last_time = time.time()
        #power calculation
        
        power = getTelnetPower(SP2_tel,power)   
        if power > max_power:
            max_power = power
        total_power += power
        mem = psutil.virtual_memory().used
        
        if mem > max_mem:
            max_mem = mem
        #sampling rate = 0.2s
        if stop():
            break
        elapsed = time.time() - last_time
        time.sleep(max(0,DELAY-elapsed))
    print("Done")



#argument parser
parser = argparse.ArgumentParser(description='EE379K project - Deployment code')
parser.add_argument('--model', type=str, default='mbvn1_e0_f5.onnx', help='Enter model name')


args = parser.parse_args()


# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(args.model)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))



SP2_tel = tel.Telnet('192.168.4.1')


# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
accuracy = 0
total_inference_time = 0
stop_thread = False
t1 = threading.Thread(target = power_thread, args =(SP2_tel, lambda : stop_thread, ))
t1.start()
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        last_time  = time.time()
        #print("Image shape:", np.float32(img).shape)
        # For PyTorch models ONLY: normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)
        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        start_time = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        total_inference_time += time.time() - start_time
        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]
        #true label
        true_class = ''.join([i for i in filename[:-4] if i.isalpha()])        
        #accuracy calculation
        if pred_class == true_class:
            accuracy += 1
stop_thread = True
t1.join()
results_dict=  {"accuracy" : accuracy/10000,
                "avg_latency":  total_inference_time/10000,
                "max_power":  max_power,
                "avg_energy": total_power*DELAY/10000,
                "max_mem": max_mem}

with open(args.model[:-5]+'.json', 'w') as f:
    f.write(json.dumps(results_dict))
    f.write('\n')

print("Accuracy: " + str(accuracy/10000))
print("Average Latency: " + str(total_inference_time/10000))
print("Maximum Power Cconsumption: " + str(max_power))
print("Maximum memory: " + str(max_mem))
print("Average Energy Consumption: " +str(total_power*DELAY/10000))
