# import torch
# from models import BaselineMLP, MITInspired, ASAN

# data = torch.load('data.pt')
# test_data, test_labels = data[2], data[3]

# models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}
# results = {}

# for name, model_class in models.items():
#     accs = []
#     for fold in range(5):
#         model = model_class()
#         model.load_state_dict(torch.load(f'{name}_fold{fold}.pt'))
#         model.eval()
#         with torch.no_grad():
#             preds = torch.argmax(model(test_data), dim=1)
#             acc = (preds == test_labels).float().mean().item()
#             accs.append(acc)
#     avg_acc = sum(accs) / len(accs)
#     results[name] = avg_acc
#     print(f'{name} Average Test Accuracy: {avg_acc:.4f}')



import torch
from models import BaselineMLP, MITInspired, ASAN

data = torch.load('data.pt')
test_data, test_labels = data[2], data[3]

models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}

for name, model_class in models.items():
    maes = []
    for fold in range(5):
        model = model_class()
        model.load_state_dict(torch.load(f'{name}_fold{fold}.pt'))
        model.eval()
        with torch.no_grad():
            preds = model(test_data).squeeze()
            mae = torch.mean(torch.abs(preds - test_labels)).item()
            maes.append(mae)
    avg_mae = sum(maes) / len(maes)
    print(f'{name} Average MAE: {avg_mae:.4f}')