#! /usr/bin/env python
# -*- coding: utf-8
'''
Python implementation of Krippendorff's alpha -- inter-rater reliability

(c)2011-17 Thomas Grill (http://grrrr.org)
(c)2024 Marcos Treviso (@mtreviso) -- added multilabel support

Python version >= 2.4 required
'''

from __future__ import print_function

try:
    import numpy as np
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


# Additional metrics for handling multilabel data
def dice_metric(a, b):
    return 1 - 2 * len(set(a) & set(b)) / (len(set(a)) + len(set(b)))


def iou_metric(a, b):
    return 1 - len(set(a) & set(b)) / len(set(a) | set(b))


def masi_metric(a, b):
    labels_inter = set(a) & set(b)
    labels_union = set(a) | set(b)
    len_labels = [len(set(a)), len(set(b))]
    if len(set(len_labels)) == 1 and len_labels[0] == len(labels_inter):
        m = 1
    elif len(labels_inter) == min(len_labels):
        m = 0.67
    elif len(labels_inter) > 0:
        m = 0.33
    else:
        m = 0
    return 1 - m * len(labels_inter) / len(labels_union)


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=lambda x: x, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):

    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items

    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: identity function)
    missing_items: indicator for missing items (list) (default: None)
    '''

    # metric compatible with numpy
    all_np_metrics = (nominal_metric, interval_metric, ratio_metric)

    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    elif isinstance(missing_items, str):
        maskitems = list(missing_items)
    else:
        maskitems = missing_items

    if np is not None and metric in all_np_metrics:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and ((metric in all_np_metrics) or force_vecmath)
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return 1. - Do / De if (Do and De) else 1.


if __name__ == '__main__':
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    # Example with nominal and interval data
    data = (
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",  # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",  # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",  # coder C
    )
    missing = ['*']  # indicator for missing values
    array = [d.split() for d in data]  # convert to 2D list of string items

    print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing, convert_items=float))
    print("interval metric: %.3f" % krippendorff_alpha(array, interval_metric, missing_items=missing, convert_items=float))

    # Example with multilabel data
    data = [
        ['[1, 2, 3]', '[1]', '[2, 3]'],
        ['[1, 2]',    '[1]', '[3, 4]'],
        ['*',         '[1]', '[3, 4]'],
    ]

    # Helper function to convert strings to sets
    str_to_set = lambda s: set(eval(s))

    print("dice metric: %.3f" % krippendorff_alpha(data, dice_metric, missing_items='*', convert_items=str_to_set))
    print("jaccard metric: %.3f" % krippendorff_alpha(data, iou_metric, missing_items='*', convert_items=str_to_set))
    print("masi metric: %.3f" % krippendorff_alpha(data, masi_metric, missing_items='*', convert_items=str_to_set))
