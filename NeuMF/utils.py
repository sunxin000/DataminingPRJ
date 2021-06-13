import torch


def save_checkpoint(model, model_dir):

    torch.save(model.state_dict(), model_dir)

def resume_checkpoint(model, model_dir):

    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)


# def use_cuda(enabled, device_id=0):
#     if enabled:
#         assert torch.cuda.is_available(), 'CUDA is not available'
#         torch.cuda.set_device(device_id)

def use_optimizer(network, params):

    if params['optimizer'] == 'sgd':
        # Stochastic Gradient Descent optimizer
        optimizer = torch.optim.SGD(network.parameters(), lr=params['sgd_lr'], momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])

    elif params['optimizer'] == 'adam':
        # Adam optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])

    elif params['optimizer'] == 'rmsprop':
        # RMSprop optimizer
        optimizer = torch.optim.RMSprop(network.parameters(), lr=params['rmsprop_lr'], alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])

    return optimizer
