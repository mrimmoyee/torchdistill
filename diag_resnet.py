import inspect
import pprint
from torchdistill.models.classification import resnet

print("resnet module members:")
members = [n for n in dir(resnet) if not n.startswith('_')]
pprint.pprint(sorted(members))

print("\nCallable factories and signatures:")
for n in sorted(members):
    obj = getattr(resnet, n)
    if callable(obj):
        try:
            print(n, inspect.signature(obj))
        except Exception:
            print(n, "<no signature>")