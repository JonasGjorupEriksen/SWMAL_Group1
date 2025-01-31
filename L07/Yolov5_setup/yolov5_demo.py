#!/usr/bin/env python3

import sys
import os
import torch

def TorchVersions():
    print("VERSIONS:")
    sysver = sys.version.replace("\n", "")
    print(f"  _sys.version                             = {sysver}")
    print(f"  torch.__version__                        = {torch.__version__}")
    print(f"  torch.cuda.is_available()                = {torch.cuda.is_available()}")
    print(f"  torch.backends.cudnn.enabled             = {torch.backends.cudnn.enabled}")
    try:
        device = torch.device("cuda")
        print(f"  torch.cuda.get_device_properties(device) = {torch.cuda.get_device_properties(device)}")
        print(f"  torch.tensor([1.0, 2.0]).cuda()          = {torch.tensor([1.0, 2.0]).cuda()}")
        return True
        
    except RuntimeError as ex:
        print(f"EXCEPTION({type(ex)}): {ex}", file=sys.stderr)
        print("WARNING: Sorry mate, it seems your PC is without a CUDA enabled GPU!", file=sys.stderr)
    except Exception as ex:
        print(f"EXCEPTION({type(ex)}): {ex}", file=sys.stderr)
    return False

def PredictDemo(figfilename):
    if not  os.path.exists(figfilename):
        raise f"file '{figfilename}' does not exists or is not readable"

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

    # Images
    #img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    img = figfilename

    # Inference
    results = model(img)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #results.show()
    results.save('temp.jpg')
    
if __name__=='__main__':
    try:
        if len(sys.argv) != 2:
            raise RuntimeError("I need a figure filename as argument")
                        
        TorchVersions()
        PredictDemo(sys.argv[1])
    except Exception as e:
        print(f"ERROR occured {e}")
        exit(-1)
