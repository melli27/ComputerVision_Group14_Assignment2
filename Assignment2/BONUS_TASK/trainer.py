import torch

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # Resets greadients from previous step
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels)
            loss.backward() # Backward pass
            optimizer.step() # Update weights

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Update learning rate at end of epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad(): # Don't calculate gradients during validation
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        avg_loss = train_loss / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Loss: {avg_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Test Acc: {test_acc:.2f}% '
              f'LR: {current_lr:.6f}')

        # save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy {best_accuracy:.2f}')
    
    print(f'\nBest Test Accuracy: {best_accuracy:.2f}%')
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
