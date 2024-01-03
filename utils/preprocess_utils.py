import torch
import torch.nn as nn
import numpy as np

class PreProcessor(nn.Module):

    def __init__(self, params):
        super(PreProcessor, self).__init__()
        
        self.params = params
        imgx, imgy = params.img_size
        
        static_features = None
        if self.params.add_landmask:
            from utils.conditioning_inputs import get_land_mask

            with torch.no_grad():
                lsm = torch.tensor(get_land_mask(params.landmask_path), dtype=torch.long)
                # one hot encode and move channels to front:
                lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
                lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))[:,:,:imgx,:imgy]

                if static_features is None:
                    static_features = lsm
                else:
                    static_features = torch.cat([static_features, lsm], dim=1)


        if self.params.add_orography:
            from utils.conditioning_inputs import get_orography

            with torch.no_grad():
                oro = torch.tensor(get_orography(params.orography_path), dtype=torch.float32)
                oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))[:,:,:imgx,:imgy]

                # normalize
                eps = 1.0e-6
                oro = (oro - torch.mean(oro)) / (torch.std(oro) + eps)

                if static_features is None:
                    static_features = oro
                else:
                    static_features = torch.cat([static_features, oro], dim=1)
        self.do_add_static_features = static_features is not None
        if self.do_add_static_feature:
            self.register_buffer("static_features", static_features, persistent=False)


    def forward(self, data):
        if self.params.add_zenith:
            # data has inp, tar, izen, tzen
            inp, tar, izen, tzen = data
            inp = torch.cat([inp, izen], dim=1)  # Concatenate input with zenith angle
        else:
            inp, tar = data

        if self.do_add_static_features:
            inp = torch.cat([inp, self.static_features], dim=1)

        return inp, tar
