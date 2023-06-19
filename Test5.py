from general_utils import load_pickled_file


class SmallestInfiniteSet:

    def __init__(self):
        self._positive_ints = [i for i in range(1, 1000)]

    def popSmallest(self) -> int:
        return self._positive_ints.pop(0)

    def addBack(self, num: int) -> None:
        for i in range(len(self._positive_ints)):
            if num < self._positive_ints[i]:
                self._positive_ints.insert(i, num)
                break
            elif num == self._positive_ints[i]:
                break


obj = SmallestInfiniteSet()
print(obj.popSmallest())
obj.addBack(1)
print(obj.popSmallest())
obj.addBack(1)
obj.addBack(2)
print(obj.popSmallest())
print(obj.popSmallest())
print(obj.popSmallest())


