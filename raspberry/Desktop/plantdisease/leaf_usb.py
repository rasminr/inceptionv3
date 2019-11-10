import os

while True:
    text= input("Press Enter to detect the leaf or press Enter after N to stop program:")
    if text=="":
        os.system("sudo python3 usb_capture.py")

        os.system("python3 label_image_fin.py --graph=/home/pi/Desktop/output_graph.pb --labels=/home/pi/Desktop/labels.txt --input_layer=Placeholder --output_layer=final_result  --image=frame0.jpg")
    else:
        break
        
