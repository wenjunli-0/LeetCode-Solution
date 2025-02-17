# coding task


# tiktok search, 一面, 2.14
# 连续子数组的最大乘积
nums = [2, 5, 8, -1, 9]
def maxProduct(nums):
    dp_pos = [1] * len(nums)
    dp_neg = [1] * len(nums)
    dp_pos[0] = nums[0]
    dp_neg[0] = nums[0]
    for i in range(1, len(nums)):
        dp_pos[i] = max(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
        dp_neg[i] = min(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
    return max(dp_pos)
print(maxProduct(nums))
