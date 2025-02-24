# tiktok search

# 2.14, 一面
# 连续子数组的最大乘积
def maxProduct(nums):
    # solution-1: space complexity: O(n)
    '''
    dp_pos = [1] * len(nums)
    dp_neg = [1] * len(nums)
    for i in range(len(nums)):
        dp_pos[i] = max(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
        dp_neg[i] = min(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
    return max(dp_pos)
    '''

    # solution-2: space complexity: O(1)
    max_prod, min_prod, res = nums[0], nums[0], nums[0]
    for i in range(len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod[i] = max(max_prod[i-1]*nums[i], nums[i])
        min_prod[i] = min(min_prod[i-1]*nums[i], nums[i])
        res = max(res, max_prod[i])
    return res


# nums = [2, 5, 8, -1, 9]
nums1 = [-2]
# print(maxProduct(nums))
print(maxProduct(nums1))


# 2.18, 二面
# 单调栈:
# 数组中每个元素的下一个更大元素问题，可以使用单调栈来实现，时间复杂度为 O(n)
def nextGreaterElement(nums):
    # n = len(nums)
    # res = [0] * n
    # stk = []
    # for i in range(n-1, -1, -1):
    #     while stk and stk[-1] <= nums[i]:
    #         stk.pop()
    #     res[i] = stk[-1] if stk else -1
    #     stk.append(nums[i])
    # return res

    n = len(nums)
    res = [0] * n
    stk = []
    for i in range(n-1, -1, -1):
        while stk and stk[-1] <= nums[i]:
            stk.pop()
        res[i] = -1 if not stk else stk[-1]
        stk.append(nums[i])
    return res


nums1 = [2, 3, 1, 6, 2, 8]
nums2 = [-2]
print(nextGreaterElement(nums1))  # 输出: [3, 6, 6, 8, 8, -1]
print(nextGreaterElement(nums2))  # 输出: [-2]


# LayerNorm
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_dim))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(hidden_dim))  # Shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean along feature dimension
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # Compute variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # Normalize
        return self.gamma * x_norm + self.beta  # Apply scale and shift


# 2.24, 三面
# 不同的二叉搜索树
def numTrees(n):
    dp = [0] * (n+1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n+1):
        for j in range(1, i+1):
            dp[i] += dp[j-1] * dp[i-j]
    return dp[n]


