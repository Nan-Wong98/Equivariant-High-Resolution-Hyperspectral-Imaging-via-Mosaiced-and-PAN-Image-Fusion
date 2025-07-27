import argparse
import torch.nn as nn
import torch
from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from model import Mpattern_opt, Generator
import time, numpy

def get_param_and_flops_of_each_layer(net, net_name, params, flops, memory):
    keys = list(net._modules.keys())
    if keys == []:
        if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ReLU) \
        or isinstance(net, torch.nn.PReLU) or isinstance(net, torch.nn.ELU) \
        or isinstance(net, torch.nn.LeakyReLU) or isinstance(net, torch.nn.ReLU6) \
        or isinstance(net, torch.nn.Linear) or isinstance(net, torch.nn.MaxPool2d) \
        or isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.BatchNorm2d) \
        or isinstance(net, torch.nn.Upsample) or isinstance(net, torch.nn.ConvTranspose2d):
            params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
            param_this_layer = str(params_num)
            if params_num // 10 ** 6 > 0:
                param_this_layer = str(round(params_num / 10 ** 6, 2)) + 'M'
            elif params_num // 10 ** 3:
                param_this_layer = str(round(params_num / 10 ** 3, 2)) + 'k'
            kernel_size = str(tuple(net.weight.shape[-2:])) if hasattr(net, 'weight') else ''
            stride = str(net.stride) if hasattr(net, 'stride') else ''
            input_shape = str(tuple(net.input_shape)) if hasattr(net, 'input_shape') else ''
            output_shape = str(tuple(net.output_shape)) if hasattr(net, 'output_shape') else ''
            module_type = str(type(net)).split('.')[-1][:-2]
            mem = str(net.__mem__ // 1e6) + "MB" if hasattr(net, '__mem__') else ''
            print('{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format(net_name, module_type, input_shape,
                                                                                  output_shape, kernel_size, stride,
                                                                                  param_this_layer,
                                                                                  flops_to_string(net.__flops__), mem))
            params.append(params_num)
            flops.append(net.__flops__)
            memory.append(net.__mem__)
    else:
        for key in keys:
            get_param_and_flops_of_each_layer(net._modules[key],
                                              net_name=net_name + '.' + key,
                                              params=params,
                                              flops=flops,
                                              memory=memory)

def input_matrix_wpn(inH, inW, msfa_size):

    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    for i in range(0,msfa_size):
        h_offset_coord[i::msfa_size, :, 0] = (i+1)/msfa_size
        w_offset_coord[:, i::msfa_size, 0] = (i+1)/msfa_size

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1, 2)
    return pos_mat

# for LSAN, PanGAN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.num_bands = 16
    args.msfa_size = 4
    args.spatial_ratio = 2
    demosaic_net = Mpattern_opt(args).cuda()
    ps_net = Generator(args).cuda()
    batch_size = 1

    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])

    # demosaic
    mosaic = torch.FloatTensor(batch_size, 1, 1020, 1104).cuda()

    model = add_flops_counting_methods(demosaic_net)
    model.eval().start_flops_count()
    t = time.time()
    with torch.no_grad():
        scale_coord_map = input_matrix_wpn(mosaic.shape[2], mosaic.shape[3], MSFA.shape[0]).to(mosaic.device)
        mosaic_up = torch.zeros(mosaic.shape[0], MSFA.shape[0]*MSFA.shape[1], mosaic.shape[2], mosaic.shape[3]).to(mosaic.device)
        for i in range(MSFA.shape[0]):
            for j in range(MSFA.shape[1]):
                mosaic_up[:, i*MSFA.shape[1]+j, i::MSFA.shape[0], j::MSFA.shape[1]] = mosaic[:, 0, i::MSFA.shape[0], j::MSFA.shape[1]]
        demosaic = demosaic_net([mosaic_up, mosaic], scale_coord_map)
    params, flops, memory = [], [], []
    print(
        '{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format('module', 'type', 'input_shape', 'output_shape',
                                                                        'kernel', 'stride', 'params', 'flops', 'mem'))
    get_param_and_flops_of_each_layer(model, '', params, flops, memory)
    demosaic_params, demosaic_flops, demosaic_mem = 0, 0, 0
    for param in params:
        demosaic_params += param
    if demosaic_params // 10 ** 6 > 0:
        demosaic_params = str(round(demosaic_params / 10 ** 6, 2)) + 'M'
    elif demosaic_params // 10 ** 3:
        demosaic_params = str(round(demosaic_params / 10 ** 3, 2)) + 'k'
    for flop in flops:
        demosaic_flops += flop
    for mem in memory:
        demosaic_mem += mem

    # pansharpening
    pan = torch.FloatTensor(batch_size, 1, 2040, 2208).cuda()

    model = add_flops_counting_methods(ps_net)
    model.eval().start_flops_count()
    with torch.no_grad():
        hrms = ps_net(demosaic, pan)
    params, flops, memory = [], [], []
    print(
        '{:<50}{:<20}{:<30}{:<25}{:<15}{:<10}{:<10}{:<10}{:<10}'.format('module', 'type', 'input_shape', 'output_shape',
                                                                        'kernel', 'stride', 'params', 'flops', 'mem'))
    get_param_and_flops_of_each_layer(model, '', params, flops, memory)
    ps_params, ps_flops, ps_mem = 0, 0, 0
    for param in params:
        ps_params += param
    if ps_params // 10 ** 6 > 0:
        ps_params = str(round(ps_params / 10 ** 6, 2)) + 'M'
    elif ps_params // 10 ** 3:
        ps_params = str(round(ps_params / 10 ** 3, 2)) + 'k'
    for flop in flops:
        ps_flops += flop
    for mem in memory:
        ps_mem += mem
    infer_time = time.time() - t

    print('Demosai_net Flops:  {}'.format(flops_to_string(demosaic_flops / batch_size)))
    print('Demosai_net Params: {}'.format(demosaic_params))
    print('Demosai_net Memory usage: {} GB'.format(demosaic_mem / batch_size / 1e9))
    print('Ps_Net Flops:  {}'.format(flops_to_string(ps_flops / batch_size)))
    print('Ps_Net Params: {}'.format(ps_params))
    print('Ps_Net Memory usage: {} GB'.format(ps_mem / batch_size / 1e9))
    print('Infer time: {}s'.format(infer_time / batch_size))