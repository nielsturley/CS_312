class BinaryHeap:
    def __init__(self):
        self.heap = []  # array of tuples (node_id, index)
        self.pointer_array = {}  # maps node_id to index in heap
        self.size = 0

    def push(self, item):
        self.heap.append(item)  # item is a tuple (node_id, index)
        self.size += 1
        tree_index = self.bubble_up(self.size - 1)
        self.pointer_array[item[0]] = tree_index

    def pop(self):
        if self.size == 0:
            return None

        # Swap the root with the last element
        root = self.heap[0]
        self.pointer_array[root[0]] = None
        self.heap[0] = self.heap[self.size - 1]
        self.pointer_array[self.heap[0][0]] = 0
        self.heap.pop()
        self.size -= 1
        self.bubble_down(0)

        return root

    def bubble_up(self, i):
        while i > 0 and self.heap[i][1] < self.heap[self.parent(i)][1]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
        return i

    def bubble_down(self, i):
        min_index = i

        # Recursively bubble down the smaller child
        l = self.left(i)
        if l < self.size and self.heap[l][1] < self.heap[min_index][1]:
            min_index = l

        r = self.right(i)
        if r < self.size and self.heap[r][1] < self.heap[min_index][1]:
            min_index = r

        if i != min_index:
            self.swap(i, min_index)
            self.bubble_down(min_index)

    def parent(self, i):
        return (i - 1) // 2

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pointer_array[self.heap[i][0]] = i
        self.pointer_array[self.heap[j][0]] = j

    def update(self, index, dist):
        # Use the pointer array to find the index in the heap, then update the heap
        tree_index = self.pointer_array[index]
        self.heap[tree_index] = (self.heap[tree_index][0], dist)
        self.bubble_up(tree_index)
        self.bubble_down(tree_index)


