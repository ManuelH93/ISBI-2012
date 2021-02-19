epoch = 1

for phase in ['train', 'val']:
            
    if phase == 'train':
        print(f'train_{epoch}')
    else:
        epoch_loss = epoch + 1
        print('hello')

print(epoch_loss)