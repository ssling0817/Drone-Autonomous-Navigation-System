#python3 detect_uav.py --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_200.pth --class_path data/custom/classes.names
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import matplotlib; matplotlib.use('agg')    
import cv2

import threading
import socket
import pyscreenshot as ImageGrab
#import keyboard
from pynput import keyboard
from pynput.keyboard import Key, Listener 
#00

cap=cv2.VideoCapture("udp://192.168.10.1:11111")	


num=1
def save_result():
    global num
    image = ImageGrab.grab()
    
    # 儲存檔案
    image.save("result/"+str(num)+".jpg")
    num=num+1
'''
class result_thread(threading.Thread):
    def __init__(self):	
        threading.Thread.__init__(self)
    def on_press(key): 
        if(key==Key.f1):
            save_result() 
    with Listener(on_press=on_press) as listener: 
        listener.start() 
res_t=result_thread()
res_t.start()
'''
def on_press(key): 
    if(key==Key.f1):
        save_result() 
listener = keyboard.Listener(on_press=on_press)
listener.start() 
#get frame in background
current_frame=None 
class cam_thread(threading.Thread):
    def __init__(self):	
        global cap
        threading.Thread.__init__(self)
    def run(self):			
        global cap			
        global current_frame			
        global current_ret			
        while True:			
            current_ret, current_frame=cap.read()			
            time.sleep(0.03)		

cam_t=cam_thread()	
cam_t.start()	

host = ''			
recv_data=""			
port = 9000		
locaddr = (host,port)			
			
# Create a UDP socket			
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)			
tello_address = ('192.168.10.1', 8889)			
sock.bind(locaddr)			
			

#hold UAV per 10 seconds
class stop_thread(threading.Thread):			
    def __init__(self):			
        #print("stop thread init")			
        threading.Thread.__init__(self)			
			
                    			
    def run(self):			
        global tello_address			
        while True:			
            print("send stop")			
            sock.sendto("stop".encode(encoding="utf-8"), tello_address)			
            time.sleep(6)			
stop_t=stop_thread()			
		
def recv():						
    global recv_data			
    while True:			
        try:			
            data, server = sock.recvfrom(1518)			
            recv_data=data.decode(encoding="utf-8")			
            print("{}: {}".format(server, recv_data))			
            			
        except Exception:			
            print ('\nExit . . .\n')			
            break			
#00
#todo: connect to UAV and open video stream
#(you can use lab1/example.py to send "streamon" to UAV first)



print ('\r\n\r\nTello Python3 Demo.\r\n')

print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')

print ('end -- quit demo.\r\n')





msg = "command"
msg = msg.encode(encoding="utf-8") 
sent = sock.sendto(msg, tello_address)	
time.sleep(3)
print('command f')

'''
msg="streamon"
msg = msg.encode(encoding="utf-8") 
sent = sock.sendto(msg, tello_address)	
time.sleep(0.1)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_194.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
        print("load custom.pth")

    model.eval()  

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Tensor = torch.FloatTensor

    
    fr = 0
    r = 1

    target="Aquarius"			
    
    detect=False
    stop_t.start()
#00
    while(cap.isOpened()==False):
        print("---------------opening cap--------------------")
        cap.open("udp://192.168.10.1:11111")
    while (cap.isOpened()):
        time.sleep(0.05)
        #fr+=1
        #ret, frame = cap.read()
        frame=current_frame
        
        if (fr % r == 0):
            
            input_imgs, _ = transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)])((frame,np.zeros((1, 5))))
            input_imgs = Variable(input_imgs.type(Tensor)).unsqueeze(0)
            #print ('---------------transforming---------------')
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            #print ('---------------Create plot---------------')
            # Create plot
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            
            #print ('---------------Detection-----------------')

            if detections is not None:
                # Rescale boxes to original image
#00
                
                detections = rescale_boxes(detections[0], opt.img_size, img.shape[:2])

                for x1, y1, x2, y2, conf, cls_pred in detections:
                    #if(classes[int(cls_pred)]!=target):
                    #    continue;
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = (0.2235294117647059, 0.23137254901960785, 0.4745098039215686, 1.0)
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )


                    
                    x_center=(x1+x2)/2
                    y_center=(y1+y2)/2
                    print("class: {}, center: ({},{})".format(classes[int(cls_pred)],x_center, y_center))
                    
                    
                    #-----

                    #todo: check target if it is in the screen 


                    #todo: send command to control UAV



                    #------
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
                img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imshow("plot",img)
            plt.close('all')

   

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#cv2.imwrite("result/"+target+str(num)+".jpg", img)

    cap.release()
    cv2.destroyAllWindows()

