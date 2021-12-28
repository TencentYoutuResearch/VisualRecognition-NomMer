# --------------------------------------------------------
# NomMer Transformer
# Created by clarkjiang
# 2021/12/27.
# --------------------------------------------------------

from .nommer import NomMerAttn

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'nommer_attn':
        model = NomMerAttn(emd_dim=config.MODEL.NomMer.EMBED_DIM,
                            depths=config.MODEL.NomMer.DEPTHS,
                            num_heads=config.MODEL.NomMer.NUM_HEADS,
                            input_size=config.DATA.IMG_SIZE,
                            win_size=config.MODEL.NomMer.WINDOW_SIZE,
                            pool_size=config.MODEL.NomMer.POOLING_SIZE,
                            cnn_expansion=config.MODEL.NomMer.CNN_EXPANSION,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            num_class=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
