class Solution:
    def canPartition(self, nums) -> bool:
        sumn = sum(nums)
        if sumn % 2 != 0:
            return False
        n = len(nums)
        sumn = sumn // 2
        dp = [False] * (sumn + 1)

        # dp
        dp[0] = True
        for i in range(n):
            for j in range(sumn, -1, -1):
                if j - nums[i] >= 0:
                    dp[j] = dp[j] or dp[j-nums[i]]
        return dp[sumn]
