from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import sys
import numpy as np
import argparse
import random
import glob

import tqdm

from defense_utils import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', default="../../dataset/N_PreFine_CW/", help='Undefended dataset')
parser.add_argument('--output_path', default="../../dataset/N_PreFine_CW_regulator/", help='Output path')
parser.add_argument('--n_processes', type=int, help='Number of python processes to run in parallel', default='4')
parser.add_argument('--orig_rate', help='Original packet surge rate', default='277')
parser.add_argument('--dep_rate', help='Packet sending depreciation rate', default='.94')
parser.add_argument('--budget', type=int, help='Maximum possible padding budget', default='2000')
parser.add_argument('--threshold', help='Burst threshold', default='3.55')
parser.add_argument('--upload_ratio', help='Ratio of download packets to upload packets', default='3.95')
parser.add_argument('--delay_cap', help='Maximum upload packet delay', default='1.77')
args = parser.parse_args()

CUTOFF_LENGTH = 20000
CUTOFF_TIME = 120
SAVE_PICKLE = False

# if not os.path.exists(args.output_path):
#     os.makedirs(args.output_path)


def regulator_download(target_trace, target_trace_l):
    orig_rate = float(args.orig_rate)
    depreciation_rate = float(args.dep_rate)
    max_padding_budget = int(args.budget)
    burst_threshold = float(args.threshold)

    padding_budget = np.random.randint(0, max_padding_budget)

    output_trace = []
    upload_trace = []

    if len(target_trace) >= 11:
        position = 10
    else:
        position = 0

    # send packets at a constant rate initially (to construct circuit)
    download_start = target_trace[position]
    added_packets = int(download_start * 10)
    for i in range(added_packets):
        pkt_time = i * .1
        output_trace.append((pkt_time, -random.randint(512, 1535)))

    output_trace.append((target_trace[position], target_trace_l[0]))
    current_time = download_start
    burst_time = target_trace[position]

    padding_packets = 0
    position = 1

    while True:
        # calculate target rate
        target_rate = orig_rate * (depreciation_rate ** (current_time - burst_time))

        if (target_rate < 1):
            target_rate = 1

            # if the original trace has been completely sent
        if (position == len(target_trace)):
            break

        # find number of real packets waiting to be sent
        queue_length = 0
        for c in range(position, len(target_trace)):
            if (target_trace[c] < current_time):
                queue_length += 1
            else:
                break

        # if waiting packets exceeds treshold, then begin a new burst
        if (queue_length > (burst_threshold * target_rate)):
            burst_time = current_time

        # calculate gap
        gap = 1 / float(target_rate)
        current_time += gap

        if (queue_length == 0 and padding_packets >= padding_budget):
            # no packets waiting and padding budget reached
            continue
        elif (queue_length == 0 and padding_packets < padding_budget):
            # no packets waiting, but padding budget not reached
            output_trace.append((current_time, -random.randint(512, 1535)))
            padding_packets += 1
        else:
            # real packet to send
            output_trace.append((current_time, target_trace_l[position]))
            position += 1

    output_trace_l = [v for k, v in output_trace]
    output_trace = [k for k, v in output_trace]

    return output_trace, output_trace_l


def regulator_upload_full(download_trace, upload_trace, upload_trace_l):
    upload_ratio = float(args.upload_ratio)
    delay_cap = float(args.delay_cap)

    output_trace = []

    # send one upload packet for every $upload_ratio download packets
    upload_size = int(len(download_trace) / upload_ratio)
    output_trace = list(np.random.choice(download_trace, upload_size))

    # send at constant rate at first
    if len(download_trace) >= 11:
        download_start = download_trace[10]
    else:
        download_start = download_trace[0]
    added_packets = int(download_start * 5)
    for i in range(added_packets):
        pkt_time = i * .2
        output_trace.append(pkt_time)

    # assign each packet to the next scheduled sending time in the output trace
    output_trace = sorted(output_trace)
    delay_packets = []
    packet_position = 0

    length_sequence = [random.randint(512, 1535) for _ in range(len(output_trace))]  # [512] * len(output_trace)
    delay_length_sequence = []
    for i, t in enumerate(upload_trace):
        found_packet = False
        for p in range(packet_position + 1, len(output_trace)):
            if (output_trace[p] >= t and (output_trace[p] - t) < delay_cap):
                packet_position = p
                found_packet = True
                length_sequence[packet_position] = upload_trace_l[i]
                break

        # cap delay at delay_cap seconds
        if (found_packet == False):
            delay_packets.append(t + delay_cap)
            delay_length_sequence.append(upload_trace_l[i])

    length_sequence += delay_length_sequence
    output_trace += delay_packets

    output_trace = [(k, v) for k, v in zip(output_trace, length_sequence)]
    output_trace = sorted(output_trace, key=lambda x: x[0])

    output_trace_l = [v for k, v in output_trace]
    output_trace = [k for k, v in output_trace]
    return output_trace, output_trace_l


def cost_calc(orig_trace, alt_trace):
    '''calculate the bandwidth and latency overhead for download traces'''
    dummy_padding = len(alt_trace) - len(orig_trace)

    latency_cost = 0.0
    sending_time = 0.0
    last_packet_sent = 0
    last_packet_latency = 0.0
    for t in orig_trace:
        # find next available packet in sending schedule
        available = alt_trace[last_packet_sent:]
        for p in available:
            if (p > t or p == t):
                sending_time = p
                last_packet_sent = alt_trace.index(p) + 1
                break

        latency_cost += (sending_time - t)
        # finds latency of last real packet
        last_packet_latency = (sending_time - t)

    return (dummy_padding, latency_cost, last_packet_latency)


def cost_calc_max_latency(orig_trace, alt_trace):
    '''calculates latency overhead for upload trace using a more pessimistic method'''
    dummy_padding = len(alt_trace) - len(orig_trace)

    latency_cost = 0.0
    sending_time = 0.0
    last_packet_sent = 0
    last_packet_latency = 0.0
    max_packet_latency = 0.0
    counter = 0
    location = 0
    for t in orig_trace:
        # find next available packet in sending schedule
        available = alt_trace[last_packet_sent:]
        for p in available:
            if (p > t or p == t):
                sending_time = p
                last_packet_sent = alt_trace.index(p) + 1
                break

        latency_cost += (sending_time - t)

        # find packet with largest delay
        if ((sending_time - t) > max_packet_latency and counter > 10):
            max_packet_latency = (sending_time - t)
            location = counter
        counter += 1
        last_packet_latency = (sending_time - t)

    return max_packet_latency, location


def simulate(filename):
    file_name = filename.split("/")[-1]

    trace = get_trace(filename, CUTOFF_TIME, CUTOFF_LENGTH)

    # get download and upload separately
    download_packets, download_packets_l = get_download_packets(trace)
    upload_packets, upload_packets_l = get_upload_packets(trace)
    original_bandwidth = len(upload_packets) + len(download_packets)

    # get defended traces
    padded_download, padded_download_l = regulator_download(download_packets, download_packets_l)
    padded_upload, padded_upload_l = regulator_upload_full(padded_download, upload_packets, upload_packets_l)
    padded_bandwidth = len(padded_download) + len(padded_upload)

    # calculate latency overhead
    _, _, download_latency_overhead = cost_calc(download_packets, padded_download)
    upload_latency_overhead, _ = cost_calc_max_latency(upload_packets, padded_upload)
    latency_overhead = download_latency_overhead + upload_latency_overhead

    download_packets = [(p, l) for p, l in zip(padded_download, padded_download_l)]
    upload_packets = [(p, l) for p, l in zip(padded_upload, padded_upload_l)]

    both_output = sorted(download_packets + upload_packets, key=lambda x: x[0])

    # output to file
    dir_name = args.output_path
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    path = os.path.join(dir_name, str(file_name))
    with open(path, 'w') as w:
        for p in both_output:
            w.write(str(p[0]) + '\t' + str(p[1]) + '\n')
            
    return trace, both_output
    
    if (SAVE_PICKLE):
        # output .pkl files
        direction_only = [np.float64(x[1]) for x in both_output]

        # pad to 5000 (for deep fingerprinting attack)
        if (len(direction_only) > 5000):
            direction_only = direction_only[:5000]
        else:
            direction_only += [0] * (5000 - len(direction_only))

        return np.asarray(direction_only), np.int64(website)


if __name__ == '__main__':
    file_list = glob.glob(os.path.join(args.source_path, "*"))

    p = Pool(args.n_processes)

    if (SAVE_PICKLE):
        all_indiv_streams = []
        all_indiv_streams = p.map(simulate, file_list)
        all_indiv_streams_real = []
        for x in all_indiv_streams:
            if x is not None:
                all_indiv_streams_real.append(x)
        all_indiv_streams = all_indiv_streams_real

        website_list = [x[1] for x in all_indiv_streams]
        trace_list = [x[0] for x in all_indiv_streams]

        output_pkl(trace_list, website_list, args.output_path)

    else:
        def latency(trace):
            if len(trace) < 2:
                return 0.0
            return trace[-1][0] - trace[0][0]


        def bandwidth(trace):
            total_bytes = sum([abs(p[1]) for p in trace])
            return 1.0 * total_bytes / latency(trace)


        def bandwidth_ovhd(new, old):
            bw_old = bandwidth(old)
            if bw_old == 0.0:
                return 0.0
            return 1.0 * bandwidth(new) / bw_old


        def latency_ovhd(new, old):
            lat_old = latency(old)
            if lat_old == 0.0:
                return 0.0
            return 1.0 * latency(new) / lat_old

        latencies, bandwidths = [], []
        for filename in tqdm.tqdm(file_list):
            trace, simulated = simulate(filename)
            bw_ovhd = bandwidth_ovhd(simulated, trace)
            bandwidths.append(bw_ovhd)

            lat_ovhd = latency_ovhd(simulated, trace)
            latencies.append(lat_ovhd)

        print("Latency overhead: %s" % np.median([l for l in latencies if l > 0.0]))
        print("Bandwidth overhead: %s" % np.median([b for b in bandwidths if b > 0.0]))

        # p.map(simulate, file_list)
