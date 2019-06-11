def param_setting_resnet(model):
    fc_params = list(map(id, model.fc.parameters()))
    layer4_1_params = list(map(id, model.layer4[1].parameters()))
    # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    base_params = filter(lambda p: id(p) not in fc_params + layer4_1_params, model.parameters())

    param_list = [{'params': base_params, 'lr': 0},
                  {'params': model.layer4[1].parameters(), 'lr': 1},
                  {'params': model.fc.parameters(), 'lr': 1}]
    # param_list = [{'params': model.parameters(), 'lr': 1}]
    return param_list

def param_setting_jointmodel(model):
    fc_params = list(map(id, model.fc.parameters())) + list(map(id, model.img_fc.parameters()))
    layer4_1_params = list(map(id, model.imgNet.layer4[1].parameters()))
    spe_params = list(map(id, model.speNet.parameters()))
    # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    base_params = filter(lambda p: id(p) not in fc_params + layer4_1_params + spe_params,
                         model.parameters())

    param_list = [{'params': base_params, 'lr': 0},
                  {'params': model.imgNet.layer4[1].parameters(), 'lr': 1},
                  {'params': model.speNet.parameters(), 'lr': 0},
                  {'params': model.img_fc.parameters(), 'lr': 1},
                  {'params': model.fc.parameters(), 'lr': 1}]
    # param_list = [{'params': model.parameters(), 'lr': 1}]
    return param_list

def param_setting_jointmodel2(model):
    # fc_params = list(map(id, model.fc.parameters()))
    # post_params = list(map(id, model.post_slc))
    # pre_img_params = list(map(id, model.pre_img))
    # pre_spe_params = list(map(id, model.pre_spe))

    param_list = [{'params': model.fc.parameters(), 'lr': 1},
                  {'params': model.post_slc.parameters(), 'lr': 1},
                  {'params': model.pre_img.parameters(), 'lr': 1}
                  # {'params': model.pre_spe.parameters(), 'lr': 0.1}
                  ]

    return param_list

def param_setting_jointdcamodel2(model):
    # fc_params = list(map(id, model.fc.parameters()))
    # post_params = list(map(id, model.post_slc))
    # pre_img_params = list(map(id, model.pre_img))
    # pre_spe_params = list(map(id, model.pre_spe))

    param_list = [{'params': model.fc.parameters(), 'lr': 1},
                  {'params': model.post_slc.parameters(), 'lr': 1}
                  # {'params': model.pre_spe.parameters(), 'lr': 0.1}
                  ]

    return param_list