# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.01)
# #     criterion = nn.CrossEntropyLoss()
# #     for epoch in range(100):
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved.')

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}
# # losses_dict = {}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.005)  # Lowered LR for ASAN stability
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR every 50 epochs
# #     criterion = nn.CrossEntropyLoss()
# #     losses = []
# #     for epoch in range(200):  # Increased to 200 epochs
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         losses.append(loss.item())
# #     losses_dict[name] = losses
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved. Final loss: {losses[-1]:.4f}')
# #     # Save losses for analysis
# #     with open(f'results/{name}_losses.txt', 'w') as f:
# #         f.write('\n'.join(map(str, losses)))

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}
# # losses_dict = {}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # Added weight_decay
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
# #     criterion = nn.CrossEntropyLoss()
# #     losses = []
# #     for epoch in range(200):
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         losses.append(loss.item())
# #     losses_dict[name] = losses
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved. Final loss: {losses[-1]:.4f}')
# #     with open(f'results/{name}_losses.txt', 'w') as f:
# #         f.write('\n'.join(map(str, losses)))

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}
# # losses_dict = {}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)  # Lower LR, higher weight_decay
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
# #     criterion = nn.CrossEntropyLoss()
# #     losses = []
# #     for epoch in range(250):  # Increased to 250 epochs
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         losses.append(loss.item())
# #     losses_dict[name] = losses
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved. Final loss: {losses[-1]:.4f}')
# #     with open(f'results/{name}_losses.txt', 'w') as f:
# #         f.write('\n'.join(map(str, losses)))


# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}
# # losses_dict = {}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)  # Reduced weight_decay
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)  # Adjusted step_size
# #     criterion = nn.CrossEntropyLoss()
# #     losses = []
# #     for epoch in range(300):  # Increased to 300 epochs
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         losses.append(loss.item())
# #     losses_dict[name] = losses
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved. Final loss: {losses[-1]:.4f}')
# #     with open(f'results/{name}_losses.txt', 'w') as f:
# #         f.write('\n'.join(map(str, losses)))



# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]

# # models = {'Baseline': BaselineMLP(), 'MIT-Inspired': MITInspired(), 'ASAN': ASAN()}
# # losses_dict = {}

# # for name, model in models.items():
# #     optimizer = optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-4)  # Slightly lower LR
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
# #     criterion = nn.CrossEntropyLoss()
# #     losses = []
# #     best_loss = float('inf')
# #     patience = 50  # Early stopping patience
# #     trigger_times = 0
# #     for epoch in range(300):
# #         optimizer.zero_grad()
# #         out = model(train_data)
# #         loss = criterion(out, train_labels)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         losses.append(loss.item())
# #         # Early stopping
# #         if loss.item() < best_loss:
# #             best_loss = loss.item()
# #             trigger_times = 0
# #         else:
# #             trigger_times += 1
# #         if trigger_times >= patience:
# #             print(f'Early stopping triggered for {name} at epoch {epoch}')
# #             break
# #     losses_dict[name] = losses
# #     torch.save(model.state_dict(), f'{name}.pt')
# #     print(f'{name} trained and saved. Final loss: {losses[-1]:.4f}')
# #     with open(f'results/{name}_losses.txt', 'w') as f:
# #         f.write('\n'.join(map(str, losses)))


# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from sklearn.model_selection import KFold
# # from models import BaselineMLP, MITInspired, ASAN

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]
# # dataset = torch.utils.data.TensorDataset(train_data, train_labels)
# # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}
# # results = {name: [] for name in models}

# # for name, model_class in models.items():
# #     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
# #         train_subset = torch.utils.data.Subset(dataset, train_idx)
# #         val_subset = torch.utils.data.Subset(dataset, val_idx)
# #         train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
# #         val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)

# #         model = model_class()
# #         optimizer = optim.Adam(model.parameters(), lr=0.0025, weight_decay=1e-4)
# #         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
# #         criterion = nn.CrossEntropyLoss()
# #         best_val_loss = float('inf')
# #         patience = 50
# #         trigger_times = 0

# #         for epoch in range(300):
# #             model.train()
# #             for batch_x, batch_y in train_loader:
# #                 optimizer.zero_grad()
# #                 out = model(batch_x)
# #                 loss = criterion(out, batch_y)
# #                 loss.backward()
# #                 optimizer.step()
# #             scheduler.step()

# #             model.eval()
# #             val_loss = 0
# #             with torch.no_grad():
# #                 for batch_x, batch_y in val_loader:
# #                     out = model(batch_x)
# #                     val_loss += criterion(out, batch_y).item()
# #             val_loss /= len(val_loader)
            
# #             if val_loss < best_val_loss:
# #                 best_val_loss = val_loss
# #                 trigger_times = 0
# #             else:
# #                 trigger_times += 1
# #             if trigger_times >= patience:
# #                 print(f'Early stopping triggered for {name} fold {fold} at epoch {epoch}')
# #                 break
        
# #         results[name].append(best_val_loss)
# #         torch.save(model.state_dict(), f'{name}_fold{fold}.pt')
# #         print(f'{name} fold {fold} trained and saved. Best val loss: {best_val_loss:.4f}')

# #     avg_loss = sum(results[name]) / len(results[name])
# #     print(f'{name} average validation loss across 5 folds: {avg_loss:.4f}')

# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from sklearn.model_selection import KFold
# # from models import BaselineMLP, MITInspired, ASAN
# # import torch.utils.data

# # data = torch.load('data.pt')
# # train_data, train_labels = data[0], data[1]
# # dataset = torch.utils.data.TensorDataset(train_data, train_labels)
# # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}
# # results = {name: [] for name in models}

# # for name, model_class in models.items():
# #     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
# #         train_subset = torch.utils.data.Subset(dataset, train_idx)
# #         val_subset = torch.utils.data.Subset(dataset, val_idx)
# #         train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)  # Increased batch size
# #         val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64)

# #         model = model_class()
# #         optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-5)  # Adjusted LR and weight_decay
# #         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
# #         criterion = nn.CrossEntropyLoss()
# #         best_val_loss = float('inf')
# #         patience = 30  # Reduced patience
# #         trigger_times = 0

# #         for epoch in range(300):
# #             model.train()
# #             for batch_x, batch_y in train_loader:
# #                 optimizer.zero_grad()
# #                 out = model(batch_x)
# #                 loss = criterion(out, batch_y)
# #                 loss.backward()
# #                 optimizer.step()
# #             scheduler.step()

# #             model.eval()
# #             val_loss = 0
# #             with torch.no_grad():
# #                 for batch_x, batch_y in val_loader:
# #                     out = model(batch_x)
# #                     val_loss += criterion(out, batch_y).item()
# #             val_loss /= len(val_loader)
            
# #             if val_loss < best_val_loss:
# #                 best_val_loss = val_loss
# #                 trigger_times = 0
# #             else:
# #                 trigger_times += 1
# #             if trigger_times >= patience:
# #                 print(f'Early stopping triggered for {name} fold {fold} at epoch {epoch}')
# #                 break
        
# #         results[name].append(best_val_loss)
# #         torch.save(model.state_dict(), f'{name}_fold{fold}.pt')
# #         print(f'{name} fold {fold} trained and saved. Best val loss: {best_val_loss:.4f}')

# #     avg_loss = sum(results[name]) / len(results[name])
# #     print(f'{name} average validation loss across 5 folds: {avg_loss:.4f}')




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import KFold
# from models import BaselineMLP, MITInspired, ASAN
# import torch.utils.data

# data = torch.load('data.pt')
# train_data, train_labels = data[0], data[1]
# dataset = torch.utils.data.TensorDataset(train_data, train_labels)
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}
# results = {name: [] for name in models}

# for name, model_class in models.items():
#     for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
#         train_subset = torch.utils.data.Subset(dataset, train_idx)
#         val_subset = torch.utils.data.Subset(dataset, val_idx)
#         train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64)

#         model = model_class()
#         optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=5e-5)  # Lower LR
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#         criterion = nn.CrossEntropyLoss()
#         best_val_loss = float('inf')
#         patience = 25  # Reduced patience
#         trigger_times = 0

#         for epoch in range(300):
#             model.train()
#             for batch_x, batch_y in train_loader:
#                 optimizer.zero_grad()
#                 out = model(batch_x)
#                 loss = criterion(out, batch_y)
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()

#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for batch_x, batch_y in val_loader:
#                     out = model(batch_x)
#                     val_loss += criterion(out, batch_y).item()
#             val_loss /= len(val_loader)
            
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 trigger_times = 0
#             else:
#                 trigger_times += 1
#             if trigger_times >= patience:
#                 print(f'Early stopping triggered for {name} fold {fold} at epoch {epoch}')
#                 break
        
#         results[name].append(best_val_loss)
#         torch.save(model.state_dict(), f'{name}_fold{fold}.pt')
#         print(f'{name} fold {fold} trained and saved. Best val loss: {best_val_loss:.4f}')

#     avg_loss = sum(results[name]) / len(results[name])
#     print(f'{name} average validation loss across 5 folds: {avg_loss:.4f}')



import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from models import BaselineMLP, MITInspired, ASAN
import torch.utils.data

data = torch.load('data.pt')
train_data, train_labels = data[0], data[1]
dataset = torch.utils.data.TensorDataset(train_data, train_labels)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

models = {'Baseline': BaselineMLP, 'MIT-Inspired': MITInspired, 'ASAN': ASAN}
results = {name: [] for name in models}

for name, model_class in models.items():
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=128)

        model = model_class()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()  # For regression
        best_val_loss = float('inf')
        patience = 25
        trigger_times = 0

        for epoch in range(300):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x).squeeze()
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    out = model(batch_x).squeeze()
                    val_loss += criterion(out, batch_y).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping triggered for {name} fold {fold} at epoch {epoch}')
                break
        
        results[name].append(best_val_loss)
        torch.save(model.state_dict(), f'{name}_fold{fold}.pt')
        print(f'{name} fold {fold} trained and saved. Best val loss: {best_val_loss:.4f}')

    avg_loss = sum(results[name]) / len(results[name])
    print(f'{name} average validation loss across 5 folds: {avg_loss:.4f}')