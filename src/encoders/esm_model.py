import torch

class Model ():
    def __init__ (self, variant, device):
        self.tags = ['gpu', 'heavy', 'encoder', 'protein']
        self.props = []

        if variant == 'legacy':
            self.name = 'esm'
        else:
            self.name = f'esm_{variant}'
            
        self.variant = variant
        self.device = device
        self.load_model(variant)
        
        
    def load_model (self, variant):
        if variant == 'legacy':
            variant = 'light'
            
        if variant == 'light':
            model_name = "esm2_t30_150M_UR50D"
            self.n_layers = 30
        elif variant == 'normal':
            model_name = "esm2_t33_650M_UR50D"
            self.n_layers = 33
        else:
            raise Exception(f'Unknown variant: {variant}')
            
        self.model, self.alphabet = \
            torch.hub.load("facebookresearch/esm:main",
                           model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device)

    def __call__ (self, samples):
        data = []
        for i,protein_seq in enumerate(samples):
            data.append((str(i), protein_seq))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens.to(self.device),
                                 repr_layers=[self.n_layers],
                                 return_contacts=False)
        token_repr_ = results["representations"][self.n_layers]
        1#token_repr = token_repr_.detach().cpu().numpy()
        return token_repr#, batch_lens, batch_tokens
        
        enc_list = []
        for i, tokens_len in enumerate(batch_lens):
            enc_list.append(
                token_repr[i, 1:tokens_len-1])
                
        return enc_list
