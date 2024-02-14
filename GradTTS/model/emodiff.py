from cond_diffusion import CondDiffusion
import torch


class SER_diffusion:
    def __init__(self, in_layer, out_layer, kernel=3, padding=1):
        self.SER = torch.nn.Conv2d(in_layer, out_layer)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.SGD()

    def forward(
            self,
            x0,
            mask,
            mu,
            t,
            emo_label=None,
            spk=None,
            psd=None,
            melstyle=None,
            align_len=None
               ):
        xt, z = self.forward_diffusion(x0, mask, mu, t)

        ser_input = torch.stack([xt, t, mu])

        ser_output = self.SER(ser_input)

        loss = self.loss(ser_output, emo_label)
        loss.backward()

        self.optim.step()



class EmoDiff:
    def __init__(self):
        pass
