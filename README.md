# Python implementation of Krippendorff's alpha – inter-rater reliability

For <a href="http://en.wikipedia.org/wiki/Inter-rater_reliability">inter-rater agreements</a> for experimental data with missing values, <a href="http://en.wikipedia.org/wiki/Krippendorff's_Alpha"><em>Krippendorff's alpha</em></a> coefficient has been established as a standard measure.
This a very general implementation in the <a href="http://python.org">Python</a> programming language, allowing the use of arbitrary metrics. It is also accelerated for some standard metrics that allow vector math (through <a href="http://numpy.scipy.org">Numerical Python</a>) -- currently nominal, interval and rational metrics.


## Usage

### Basic Usage

To calculate Krippendorff's alpha for your data, follow these steps:

1. Prepare your data in the following formats:
   - List of dictionaries where each dictionary represents a coder's ratings for units.
   - A sequence of sequences (lists, numpy arrays, masked arrays) where rows correspond to coders and columns to items.

2. Choose a metric function (`nominal_metric`, `interval_metric`, `ratio_metric`, or define a custom one) based on the type of your data.

3. Call the `krippendorff_alpha` function with your data and chosen metric:
   ```python
   from krippendorff_alpha import krippendorff_alpha, nominal_metric

   data = [
       {'unit1': value, 'unit2': value, ...},  # coder 1
       {'unit1': value, 'unit3': value, ...},  # coder 2
       ...
   ]

   alpha = krippendorff_alpha(data, metric=nominal_metric)
   print(f"Krippendorff's alpha: {alpha:.3f}")
   ```

Missing values are handled via `missing_items` argument. You can use `convert_items` to convert an item to a different data type.



### Multilabel Support

This implementation also supports multilabel data using specific metrics such as Jaccard index, Dice coefficient, and [MASI](https://aclanthology.org/L06-1392/) coefficient:

1. Prepare your multilabel data as a list of label sets (each item is a list of labels formatted as strings).

2. Define the metric function (`iou_metric`, `dice_metric`, `masi_metric`, or define a custom one) for multilabel comparison.

3. Convert your label sets appropriately using the `convert_items` function.

4. Call the `krippendorff_alpha` function with your multilabel data and chosen metric:
   ```python
   from krippendorff_alpha import krippendorff_alpha, iou_metric

   data = [
       ['[label1, label2, ...]', '[label1]', '[label2, label3]'],  # coder 1
       ['[label1, label2]', '[label1]', '[label3, label5]'],        # coder 2
       ...
   ]

   alpha = krippendorff_alpha(data, metric=iou_metric, convert_items=lambda x: set(eval(x)))
   print(f"Krippendorff's alpha (multilabel): {alpha:.3f}")
   ```

---

Replace `[label1, label2, ...]` with your actual labels formatted as strings in your multilabel data. This section provides a guide on how to integrate and use your implementation for both standard and multilabel datasets. 

For more examples, see `krippendorff_alpha.py`.
