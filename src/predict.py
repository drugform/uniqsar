import torch

from model import Model    

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    batch_size = None
    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    m = Model.load(model_name, device)
    m.predict_file(input_file, output_file, batch_size)

