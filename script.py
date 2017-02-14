import tick_array_mod
import numpy as np
import sys

a1 = np.zeros(100)
a2 = np.ones(100)

print("a1-refcnt", sys.getrefcount(a1))
print("a2-refcnt", sys.getrefcount(a2))

print("a1-sum", a1.sum())
print("a2-sum", a2.sum())

tick_array_mod.example_array_dot(a1, a2)

print("a1-refcnt", sys.getrefcount(a1))
print("a2-refcnt", sys.getrefcount(a2))

print("a1-sum", a1.sum())
print("a2-sum", a2.sum())

a1.fill(2.0)

print("a1-sum", a1.sum())
print("a2-sum", a2.sum())

tick_array_mod.example_array_dot(a1, a2)

print("a1-sum", a1.sum())
print("a2-sum", a2.sum())

print(tick_array_mod.example_return())