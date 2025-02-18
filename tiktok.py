# tiktok search

# 2.14, 一面
# 连续子数组的最大乘积
def maxProduct(nums):
    dp_pos = [1] * len(nums)
    dp_neg = [1] * len(nums)
    dp_pos[0] = nums[0]
    dp_neg[0] = nums[0]
    for i in range(1, len(nums)):
        dp_pos[i] = max(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
        dp_neg[i] = min(dp_pos[i-1] * nums[i], dp_neg[i-1] * nums[i], nums[i])
    return max(dp_pos)


nums = [2, 5, 8, -1, 9]
print(maxProduct(nums))


# 2.18, 二面
# 数组中每个元素的下一个更大元素问题，可以使用单调栈来实现，时间复杂度为 O(n)
def next_greater_elements(nums):
    n = len(nums)  # Determine the length of the input list.
    result = [-1] * n  # Initialize the result list with -1s.
    stack = []  # Initialize an empty stack to keep track of potential next greater elements.

    # Traverse the list from right to left.
    for i in range(n - 1, -1, -1):
        # Remove elements from the stack that are less than or equal to the current element.
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        # If the stack is not empty, the top element is the next greater element for nums[i].
        if stack:
            result[i] = stack[-1]
        # Push the current element onto the stack.
        stack.append(nums[i])
    return result


def nextGreaterElement(nums):
    n = len(nums)
    res = [-1] * n
    stk = []
    for i in range(n-1, -1, -1):
        while stk and stk[-1] <= nums[i]:
            stk.pop()
        if stk:
            res[i] = stk[-1]
        stk.append(nums[i])
    return res


nums = [2, 3, 1, 6, 2, 8]
print(next_greater_elements(nums))  # 输出: [3, 6, 6, 8, 8, -1]
print(nextGreaterElement(nums))  # 输出: [3, 6, 6, 8, 8, -1]


