class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode | None) -> ListNode | None:
        if not head:
            return None

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head

        while current:
            # 检查当前节点是否有重复值
            while current.next and current.val == current.next.val:
                current = current.next  # 跳过重复节点

            # 如果 prev 的下一个节点是 current，说明没有重复，移动 prev
            if prev.next == current:
                prev = prev.next
            else:
                # 有重复，跳过所有重复节点
                prev.next = current.next

            current = current.next  # 移动到下一个节点

        return dummy.next


# 输入：head = [1, 2, 6, 6, 4, 4, 5]
# 输出：[1, 2, 5]