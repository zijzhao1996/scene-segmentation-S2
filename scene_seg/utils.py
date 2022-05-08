import re
import numpy as np
import functools
import collections
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret



############## Data parallel ##################
def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda()
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj


def dict_gather(outputs, target_device, dim=0):
    """
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU), with dictionary support.
    """
    def gather_map(outputs):
        out = outputs[0]
        if torch.is_tensor(out):
            # MJY(20180330) HACK:: force nr_dims > 0
            if out.dim() == 0:
                outputs = [o.unsqueeze(0) for o in outputs]
            return Gather.apply(target_device, dim, *outputs)
        elif out is None:
            return None
        elif isinstance(out, collections.Mapping):
            return {k: gather_map([o[k] for o in outputs]) for k in out}
        elif isinstance(out, collections.Sequence):
            return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)


class DictGatherDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return dict_gather(outputs, output_device, dim=self.dim)


class UserScatteredDataParallel(DictGatherDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        assert len(inputs) == 1
        inputs = inputs[0]
        inputs = _async_copy_stream(inputs, device_ids)
        inputs = [[i] for i in inputs]
        assert len(kwargs) == 0
        kwargs = [{} for _ in range(len(inputs))]

        return inputs, kwargs


def user_scattered_collate(batch):
    return batch


def _async_copy(inputs, device_ids):
    nr_devs = len(device_ids)
    assert type(inputs) in (tuple, list)
    assert len(inputs) == nr_devs

    outputs = []
    for i, dev in zip(inputs, device_ids):
        with cuda.device(dev):
            outputs.append(async_copy_to(i, dev))

    return tuple(outputs)


def _async_copy_stream(inputs, device_ids):
    nr_devs = len(device_ids)
    assert type(inputs) in (tuple, list)
    assert len(inputs) == nr_devs

    outputs = []
    streams = [_get_stream(d) for d in device_ids]
    for i, dev, stream in zip(inputs, device_ids, streams):
        with cuda.device(dev):
            main_stream = cuda.current_stream()
            with cuda.stream(stream):
                outputs.append(async_copy_to(i, dev, main_stream=main_stream))
            main_stream.wait_stream(stream)

    return outputs

##################### Replicate #########################
class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)