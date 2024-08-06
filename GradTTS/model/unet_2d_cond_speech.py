from GradTTS.model.diffusion import *
from pathlib import Path

class UNet2DCondSpeechModel:
    def __init__(self, in_out, att_type, dim, dims, att_dim, heads):
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.num_resolutions = len(in_out)

        # Set unet
        in_out = list(zip(dims[:-1], dims[1:]))  # [(3, 64), (64, 128), (128, 256)]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (self.num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                # MultiAttention3->frame2frame; MultiAttention2->bin2frame (simFrame2fame?)
                # Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                #                MultiAttention(dim_out, self.enc_hid_dim, att_dim, heads, dim))),
                Residual(Rezero(LinearAttention(dim_out) if att_type == "linear" else
                                MultiAttention2(dim_out, self.enc_hid_dim, att_dim, heads))),

                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        # self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
        #                                MultiAttention(mid_dim, self.enc_hid_dim, att_dim, heads)))
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim) if att_type == "linear" else
                                        MultiAttention2(mid_dim, self.enc_hid_dim, att_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                # Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                #                MultiAttention(dim_in, self.enc_hid_dim, att_dim, heads, dim / 2 if is_last else dim))),
                Residual(Rezero(LinearAttention(dim_in) if att_type == "linear" else
                                MultiAttention2(dim_in, self.enc_hid_dim, att_dim, heads))),
                Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self,
                x,
                mask,
                hids,
                t,
                attn_mask=None,
                show_attnmap=False,
                show_inout=False
                ):
        input_x = torch.clone(x)

        hiddens = []
        mask = mask.unsqueeze(1)  # x mask
        masks = [mask]
        attn_masks = [attn_mask]  # attn mask

        # Down
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)  # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)  # -> (b, c_out_i, d, l)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, hids, hids,
                                                                         attn_mask=attn_mask)
            # -> (b, c_out_i, d, l), (b, h, l_q * d_q, l_k)
            hiddens.append(x)

            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])

            if attn_mask is not None:
                attn_mask = attn_mask[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_masks.append(attn_mask)

            # show Attn image
            if show_attnmap:
                show_attn_img(attn_map, t[0].detach().cpu().numpy())

        masks = masks[:-1]
        mask_mid = masks[-1]
        attn_masks = attn_masks[:-1]
        attn_mask_mid = attn_masks[-1]

        # Mid
        x = self.mid_block1(x, mask_mid, t)
        if self.att_type == "linear":
            x, attn_map = self.mid_attn(x)
        else:
            x, attn_map = self.mid_attn(x, hids, hids, attn_mask=attn_mask_mid)
        x = self.mid_block2(x, mask_mid, t)

        # Up
        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            attn_mask_up = attn_masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x, attn_map = attn(x) if self.att_type == "linear" else attn(x, hids, hids,
                                                                         attn_mask=attn_mask_up)
            x = upsample(x * mask_up)

        # Check mid value
        if show_inout:
            show_attn_out(attn_map, x, input_x)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)
        return output, attn_map
    def forward_mix(self,
                    x,
                    mask,
                    hids1,
                    hids2,
                    t,
                    attn_mask_hids1,
                    attn_mask_hids2
                    ):

        input_x = torch.clone(x)

        hiddens = []
        mask = mask.unsqueeze(1)  # x mask
        masks = [mask]
        attn_masks_hids1 = [attn_mask_hids1]  # attn mask
        attn_masks_hids2 = [attn_mask_hids2]

        # Down
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)  # (b, c, d, l) -> (b, c_out_i, d, l)
            x = resnet2(x, mask_down, t)  # -> (b, c_out_i, d, l)
            x1, attn_map1 = attn(x) if self.att_type == "linear" else attn(x, hids1, hids1,
                                                                         attn_mask=attn_mask_hids1)
            x2, attn_map2 = attn(x) if self.att_type == "linear" else attn(x, hids2, hids2,
                                                                           attn_mask=attn_mask_hids2)
            x = x1 + x2

            # -> (b, c_out_i, d, l), (b, h, l_q * d_q, l_k)
            hiddens.append(x)

            x = downsample(x * mask_down)  # -> (b, c_out_i, d/2, l/2)
            masks.append(mask_down[:, :, :, ::2])

            attn_mask_hids1 = attn_mask_hids1[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_mask_hids2 = attn_mask_hids2[:, :, ::4, :]  # Q: d/2 * l/2 = d*l*1/4   KV: No changed
            attn_masks_hids1.append(attn_mask_hids1)
            attn_masks_hids2.append(attn_mask_hids2)

            #### show Attn image ####
            if ATTN_MAP_SHOW_FLAG:
                show_attn_img(attn_map1, t[0].detach().cpu().numpy())
                ATTN_MAP_SHOW_FLAG = False

        masks = masks[:-1]
        mask_mid = masks[-1]
        attn_mask_hids1 = attn_mask_hids1[:-1]
        attn_mask_hids2 = attn_mask_hids2[:-1]

        attn_mask_mid_hids1 = attn_mask_hids1[-1]
        attn_mask_mid_hids2 = attn_mask_hids2[-1]

        # Mid
        x = self.mid_block1(x, mask_mid, t)
        x1, attn_map1 = self.mid_attn(x, hids1, hids1, attn_mask=attn_mask_mid_hids1)
        x2, attn_map2 = self.mid_attn(x, hids2, hids2, attn_mask=attn_mask_mid_hids2)
        x = x1 + x2
        x = self.mid_block2(x, mask_mid, t)

        # Up
        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            mask_up = masks.pop()
            attn_mask_up1 = attn_mask_hids1.pop()
            attn_mask_up2 = attn_mask_hids2.pop()

            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x1, attn_map1 = attn(x) if self.att_type == "linear" else attn(x, hids1, hids1,
                                                                         attn_mask=attn_mask_up1)
            x2, attn_map2 = attn(x) if self.att_type == "linear" else attn(x, hids2, hids2,
                                                                         attn_mask=attn_mask_up2)
            x = x1 + x2
            x = upsample(x * mask_up)

        # Check mid value
        CHECK_ATTN_OUT = False
        if CHECK_ATTN_OUT:
            show_attn_out(attn_map1, x, input_x)
            #show_attn_out(attn_map2, x, input_x)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)
        return output


def show_attn_img(attn_map, t_show):
    # Show each p2f attention of each phoneme on each head
    head_n = attn_map.shape[1]
    for h in range(head_n):
        if attn_map.shape[2] % 80 != 0:
            raise IOError("attn_map shape should be divide by 80: {}".format(attn_map.shape))
        phoneme_num = int(attn_map.shape[2] / 80)
        for phoneme_index in range(phoneme_num):
            attn_np = attn_map.detach().cpu().numpy()
            p2f_attn = attn_np[0, h, phoneme_index * 80: (phoneme_index + 1) * 80, :]
            attn_time_dir = "temp/attn_time{}".format(t_show)
            if not Path(attn_time_dir).is_dir():
                Path(attn_time_dir).mkdir(parents=True, exist_ok=True)
            plt.imsave(attn_time_dir + "/head{}_phone{}.png".format(h, phoneme_index),
                       p2f_attn)


def show_attn_out(attn_map, x, input_x):
    """
    show input/output sample during unet process, and attn map of last attn layer
    """
    attn_map_show = attn_map.transpose(2, 3).detach().cpu().numpy()
    output_show = x.detach().cpu().numpy()
    attnin_show = input_x.detach().cpu().numpy()
    save_plot(attn_map_show[0, 0, :, 8::80],
              "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_map", 0),
              size=(12, 12))
    save_plot(output_show[0, 0, :, :],
              "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_out", 0))
    save_plot(attnin_show[0, 0, :, :],
              "/home/rosen/Project/Speech-Backbones/GradTTS/{}_channel{}.png".format("attn_in", 0))
