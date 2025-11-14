import sys, inspect
import transformers
from transformers import TrainingArguments

print("PY:", sys.executable)
print("transformers version:", transformers.__version__)
print("transformers location:", transformers.__file__)
print("TrainingArguments signature:")
print(inspect.signature(TrainingArguments.__init__))
