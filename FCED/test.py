from model import BertED
model = BertED(11, False) # define model
parameters = [param for param in model.input_map.parameters()]
print(parameters)