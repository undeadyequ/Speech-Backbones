import torch
import torchvision.transforms.functional as F
from GradTTS.model.base import BaseModule
from GradTTS.model.diffusion import *
from GradTTS.model.estimator import GradLogPEstimator2dCond, DiTMocha, STDit3

class CondDiffusion(BaseModule):
    """
    Conditional Diffusion that denoising mel-spectrogram from latent normal distribution with Unet or Dit core
    """
    def __init__(self,
                 n_feats,
                 dim,
                 n_spks=1,
                 gStyle_dim=80,
                 beta_min=0.05,
                 beta_max=20,
                 stdit_config=None):
        """

        Args:
            n_feats:
            dim:   mel dim
            n_spks:
            spk_emb_dim:
            beta_min:
            beta_max:
            pe_scale:
            att_type:  "linear", "crossatt"
        """
        super(CondDiffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.beta_min = beta_min
        self.beta_max = beta_max

        # estimator
        stdit_config_ext = dict(gStyle_channels=gStyle_dim, **stdit_config)
        self.estimator = STDit3(**stdit_config_ext)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def forward(self,
                z,
                mask,
                mu,
                n_timesteps,
                stoc=False,
                melstyle=None,
                emo_label=None,
                attn_mask=None,
                q_seq_dur=None,
                k_seq_dur=None,
                refenh_ind_dur=None,
                synenh_ind_dur=None
                ):
        """
        Given z, mu and conditioning (emolabel, psd, spk), denoise melspectrogram by ODE or SDE solver (stoc)
        mel_len = 100 for length consistence
        Args:
            z:      (b, 80, mel_len)
            mask:   (b, 1, mel_len)
            mu:     (b, 80, mel_len)
            n_timesteps: int
            stoc:   bool, default = False
            spk:    (b, spk_emb) if exist else  NONE
            emo:    (b, emo_emb) if exist else  NONE
            psd:    (b, psd_dim, psd_len)   psd_len is not mel_len, psd_dim = 3 when eng/pitch/dur, psd=256 when wav2vector
            melstyle: (b, ?, ?)
            emo_label: (b, 1, emo_n)
        Returns:
        """
        h = 1.0 / n_timesteps
        xt = z * mask

        self_attns_list = []
        cross_attns_list = []
        attn_img_time_ind = (0, 10, 20, 30, 40, 49)

        for i in range(n_timesteps):
            #print("{}th time".format(i))
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)

            # t: (b, ), xt: (b, mel_dim, cut_l), mask: (b, 1, cut_l), mu: (b, mel_dim, cut_l), emo_label:(b, emo_dim), melstyle:(b, mel_dim, cut_l)
            score_emo, attn_crosses = self.estimator(
                t=t, x=xt, mask=mask, mu=mu, c=emo_label, r=melstyle, attnCross=attn_mask, q_seq_dur=q_seq_dur,  k_seq_dur=k_seq_dur,
            refenh_ind_dur=refenh_ind_dur, synenh_ind_dur=synenh_ind_dur)
            attn_selfs = None

            # Get dxt
            dxt_det = 0.5 * (mu - xt) - score_emo
            if i in attn_img_time_ind:
                self_attns_list.append(attn_selfs)
                cross_attns_list.append(attn_crosses)

            ## adds stochastic term
            if stoc:
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - score_emo)
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask

        #self_attns = torch.stack(self_attns_list, dim=0)
        cross_attns = torch.stack(cross_attns_list, dim=0)
        return xt, self_attns_list, cross_attns


    def compute_loss(self,
                     x0,
                     mask,
                     mu,
                     offset=1e-5,
                     melstyle=None,
                     emo_label=None,
                     attnCross=None,
                     q_seq_dur=None,
                     k_seq_dur=None
                     ):
        """

        Args:
            x0,       # (b, mel_dim, cut_l)
            mask,  # (b, )
            mu,    # (b, mel_dim, cut_l)
            offset,
            spk: # (b, )
            psd: # (b, )
            melstyle:  # (b, mel_dim, cut_l)
            emo_label:      # (b, )
        Returns:
        """
        t = torch.rand(x0.shape[0],
                       dtype=x0.dtype,
                       device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        xt, z = self.forward_diffusion(x0, mask, mu, t)  #
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        # Concatenate condition and input
        # t: (b, ), xt: (b, mel_dim, cut_l), mask: (b, 1, cut_l), mu: (b, mel_dim, cut_l), emo_label:(b, emo_dim), melstyle:(b, mel_dim, cut_l)
        noise_estimation, attn_crosses = self.estimator(t=t, x=xt, mask=mask, mu=mu, c=emo_label, r=melstyle,
                                                        attnCross=attnCross, q_seq_dur=q_seq_dur,  k_seq_dur=k_seq_dur)
        attn_selfs = None
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt, attn_selfs, attn_crosses

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))