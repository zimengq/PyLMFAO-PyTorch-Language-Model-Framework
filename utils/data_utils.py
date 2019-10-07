#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/


def preprocess(words):
    words = words.replace('~', '').replace('`', '').replace('!', '').replace('@', '').replace('#', ''). \
        replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('*', ''). \
        replace('(', '').replace(')', '').replace('-', '').replace('_', '').replace('+', ''). \
        replace('=', '').replace('{', '').replace('}', '').replace('[', '').replace(']', ''). \
        replace('|', '').replace('\'', '').replace(':', '').replace(';', '').replace('"', ''). \
        replace('\"', '').replace('<', '').replace('>', '').replace(',', '').replace('.', ''). \
        replace('?', '').replace('/', '')
    words = [word.lower() for word in words.split()]
    return words


def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    caps = [['<s>'] + ''.join(line.split(',')[3:]).lower().split() + ['</s>'] for line in lines[1:]]
    return caps