import torch
from torchvision.utils import save_image

from segment.utils.general import get_config_from_file, initialize_from_config
from pathlib import Path
from segment.dataloader.ddr import DDRSegTrain,DDRSegEval

def load_model(config, model_path):
    # Load configuration
    config = get_config_from_file(Path("configs")/(config+".yaml"))
    # Build model
    model = initialize_from_config(config.model)
    if model_path:
         # Load model weights
        model.load_state_dict(torch.load(model_path))
    model.eval()  # set model to eval mode

    return model

def inference(model, dataloader):
    # Create a list to store output
    outputs = []
    with torch.no_grad():
        for data in dataloader:
            # Ensure data is in the right format, and move to the correct device
            data = data.to(next(model.parameters()).device)
            output = model(data)
            outputs.append(output)
    return outputs

if __name__ == '__main__':
    model= load_model(config='1channel/ddr_dr_single_unet', model_path='')
    train_dl = DDRSegEval(size=1024,seg_object='dr_single')

    for idx, i in enumerate(train_dl):
        input = torch.unsqueeze(i['label'],dim=0)
        input = torch.unsqueeze(input,dim=0)
        input = input.float()
        output = model(input)
        output = torch.sigmoid(output)
        output = torch.argmax(output,dim=1)
        # Save the input image
        save_image(input, f'./data/DDR/infer_mask/gt/input_{idx}.png')

        # Save the output image
        save_image(output.float(), f'./data/DDR/infer_mask/pred/output_{idx}.png')
        break