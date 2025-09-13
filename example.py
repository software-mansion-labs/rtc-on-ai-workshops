import torch


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x)


model = SimpleModel().eval()
inputs = (torch.ones(2),)
outputs = model(*inputs)

print(f"Model outputs: {outputs}")


from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

exported_program = export(model, inputs)
print(exported_program)

executorch_program = to_edge_transform_and_lower(
    exported_program, partitioner=[XnnpackPartitioner()]
).to_executorch()

print(executorch_program.executorch_program)

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

from executorch.runtime import Runtime

runtime = Runtime.get()

input_tensor = torch.ones(2)
program = runtime.load_program("model.pte")
method = program.load_method("forward")

outputs = method.execute([input_tensor])
print(f"Runtime outputs: {outputs[0]}")
