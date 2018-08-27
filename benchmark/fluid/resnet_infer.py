from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

import os


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet_imagenet', 'resnet_cifar10'],
        default='resnet_imagenet',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='The minibatch size.')
    parser.add_argument(
        '--log_dir', '-f', type=str, default='./',
        help='The path of the log file.')
    parser.add_argument(
        '--use_real_data', action='store_true',
        help='If set, use real data instead of fake data.')
    parser.add_argument(
        '--skip_batch_num', type=int, default=5,
        help='The first num of minibatch num to skip, for better performance test.')
    parser.add_argument(
        '--iterations', type=int, default=0,
        help='The number of minibatches.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='flowers',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--profile', action='store_true', help='If set, do profiling.')
    parser.add_argument(
        '--infer_model_path',
        type=str,
        default='./output/pass0',
        help='The directory for loading inference model (default: %(default)s).')

    args = parser.parse_args()
    return args


def user_data_reader(data):
    '''
    Creates a data reader for user data.
    '''

    def data_reader():
        while True:
            for b in data:
                yield b
    return data_reader


def infer(args):
    if not os.path.exists(args.infer_model_path):
        raise IOError("Invalid inference model path!")

    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
    else:
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]

    fake_data = [(np.random.rand(dshape[0] * dshape[1] * dshape[2]).
                  astype(np.float32), np.random.randint(1, class_dim))
                 for _ in range(1)]

    image = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # load model
    [infer_program, feed_dict,
     fetch_targets] = fluid.io.load_inference_model(args.infer_model_path, exe)
    """
    '''
      use transpiler to fuse the ops
    '''
    
    infer_prog = infer_program.clone()
    t = fluid.InferenceTranspiler()
    t.transpile(infer_prog, place)
    print(infer_prog) #show the compute graph
    """
    # infer data read

    if args.use_real_data:
        if args.data_set == 'cifar10':
            infer_reader = paddle.batch(
                paddle.dataset.cifar.test10(),
                batch_size=args.batch_size)
        elif args.data_set == 'imagenet':
            infer_reader = paddle.batch(
                reader.test(), batch_size=args.batch_size)
        else:
            infer_reader = paddle.batch(
                paddle.dataset.flowers.test(),
                batch_size=args.batch_size)
    else:
        infer_reader = paddle.batch(
            user_data_reader(fake_data),
            batch_size = args.batch_size)

    infer_accuracy = fluid.metrics.Accuracy()
    iters = 0
    batch_times = []

    for data in infer_reader():
        if iters == args.skip_batch_num:
            profiler.reset_profiler()
        if args.iterations and iters == args.iterations + args.skip_batch_num:
            break
        image = np.array(map(lambda x: x[0].reshape(dshape),
                                data)).astype("float32")
        label = np.array(map(lambda x: x[1], data)).astype("int64")
        label = label.reshape([-1, 1])

        # start to count the time
        start = time.time()
        acc, weight = exe.run(infer_program,
                      feed={feed_dict[0]:image,feed_dict[1]:label},
                      fetch_list=fetch_targets)
        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)
        infer_accuracy.update(value=acc, weight=weight)
        infer_acc = infer_accuracy.eval()
        iters += 1

        if iters <= args.skip_batch_num:
            print("Warm-up itaration")

        print("Iteration: %d, accuracy: %f, latency: %.5f s, fps: %f" %
              (iters,  np.mean(infer_acc), batch_time, fps))

    # Postprocess benchmark data
    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    # Benchmark output
    print('\nTotal examples (incl. warm-up): %d' % (iters * args.batch_size))
    print('average latency: %.5f, 99pc latency: %.5f' %
            (latency_avg, latency_pc99))
    print('average fps: %.5f s, fps for 99pc latency: %.5f' %
            (fps_avg, fps_pc99))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler(args.device, sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)
