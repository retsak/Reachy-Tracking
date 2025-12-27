import inspect
from reachy_sdk import ReachySDK

print("Results:")
print(inspect.signature(ReachySDK.__init__))
print("Done.")
