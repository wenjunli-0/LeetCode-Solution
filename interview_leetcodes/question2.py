# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root) -> bool:
        return self._isValidBST(root, None, None)

    def _isValidBST(self, root, min, max):
        if root is None:
            return True
        if min is not None and root.val <= min.val:
            return False
        if max is not None and root.val >= max.val:
            return False
        return self._isValidBST(root.left, min, root) and self._isValidBST(root.right, root, max)
        