#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run_test.py

import os, sys
import glob
import subprocess
import re

EXEC = './image-stitching'
THRESHOLD = 0.8

def good_size(x_test, x_truth):
    ratio = x_test * 1.0 / x_truth
    if ratio > 1:
        ratio = 1.0 / ratio
    return ratio > THRESHOLD

def test_final_size(image_globs, w, h):
    #print "Testing with {}".format(image_globs)
    images = sorted(glob.glob(image_globs))
    #print ([EXEC] + images)

    cmd = [EXEC] + images
    outputs = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    outputs = outputs.split('\n')
    #print '\n'.join(outputs)
    for line in outputs:
        if 'Final Image Size' in line:
            m = re.match(r'.*\(([0-9]+), ([0-9]+)\)', line)
            ww, hh = map(int, m.group(1, 2))
            if good_size(ww, w) and good_size(hh, h):
                return
            break
    #print "Test Failed!"
    sys.exit(1)

if __name__ == '__main__':
    test_final_size('example-data/zijing/*', 6488, 1100)
    #test_final_size('example-data/CMU1/*', 8000, 1449)
    print "Tests Passed"
