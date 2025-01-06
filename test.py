# 1. 给定一个无序的整数数组，找到其中最长递增子序列的长度。例如，对于输入数组 [10, 9, 2, 5, 3, 7, 101, 18]，输出为 4，因为最长递增子序列为 [2, 3, 7, 101]。
# 2. 两个(具有不同单词的)文档的交集(intersection)中元素的个数除以并集(union)中元素的个数，就是这两个文档的相似度。

test = [
  [14, 15, 100, 9, 3],
  [32, 1, 9, 3, 5],
  [15, 29, 2, 6, 8, 7],
  [7, 10]
]

# [
#   "0,1: 0.2500",
#   "0,2: 0.1000",
#   "2,3: 0.1429"
# ]
import numpy as np

def findSimilarity(ary):
    res = []
    for idx1, list1 in enumerate(ary):
        for idx2, list2 in enumerate(ary):
            if list1 == list2:
                continue

            # intersection
            intersects = set()
            for num1 in list1:
                for num2 in list2:
                    if num1 == num2:
                        intersects.add(num1)
            # union
            union = set(list1 + list2)

            sim = len(intersects) / len(union)
            if sim > 0:
                sim = np.round(sim, 4)
                res.append([idx1, idx2, sim])
    return res


res = findSimilarity(test)
print(f'{res}')
