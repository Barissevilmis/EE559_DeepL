import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

print('train_input', train_input.size(), 'train_target', train_target.size())
print('test_input', test_input.size(), 'test_target', test_target.size())
