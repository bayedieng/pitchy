import torch 
import torchcrepe
import os 
import onnx 
crepe = torchcrepe.Crepe("full")
crepe.load_state_dict(torch.load("crepe-full.pth"))
crepe.eval()

# test input 
audio, sr = torchcrepe.load.audio("c_note.wav")
hop_length = int(sr / 200.)
time = audio.shape[1]
x = torch.randn((1 + int(time // hop_length), 1024))
if not os.path.exists("crepe-full.pth"):
    torch.onnx.export(crepe,        
                    x,                        
                    "crepe.onnx",  
                    export_params=True,       
                    opset_version=14,         
                    do_constant_folding=True, 
                    input_names = ['input'],  
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},   
                                    'output' : {0 : 'batch_size'}})
    print("Pytorch model converted to onnx.")

onnx_model = onnx.load("crepe.onnx")
onnx.checker.check_model(onnx_model)