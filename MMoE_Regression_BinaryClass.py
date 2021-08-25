from data_process import *
import MMoE


def train():
    # Sets hyper-parameters
    lr = 1e-4
    n_epochs = 50
    tasks = 11
    patience = 10
    # BATCH_SIZE=4096

    # # Defines loss function and optimizer
    loss_fn = nn.BCELoss(reduction='mean')
    early_stopping = EarlyStopping(patience, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_weight = [0, 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56, 0.63, 0.3]

    losses = []
    val_losses = []

    watch_auc = []
    share_auc = []
    sum_auc = []

    # Training loop
    for epoch in range(n_epochs):
        model.train()

        # Uses loader to fetch one mini-batch for training
        epoch_loss = []
        c = 0
        print("\nEpoch: {}/{}".format(epoch, n_epochs)) 
        for x_batch, y_batch in train_loader:

            # NOW, sends the mini-batch data to the device
            # so it matches location of the MODEL
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # One stpe of training
            yhat = model(x_batch.float())

            loss = 0
            for i in range(tasks):
                loss += loss_fn(yhat[:,i].float(), y_batch[:, i].view(-1, 1).float()) * loss_weight[i]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            c += 1
            if c % 100 == 0:     
                mesg = f'[Batch: {c}]' + \
                         f'train_loss: {np.mean(loss.item()):.5f} '
                logging.info(mesg)
        losses.append(np.mean(epoch_loss))

        # After finishing training steps for all mini-batches,
        # it is time for evaluation!

        # We tell PyTorch to NOT use autograd...
        with torch.no_grad():
            # Uses loader to fetch one mini-batch for validation
            epoch_loss = []
            epoch_watch_auc = []
            epoch_share_auc = []
            epoch_sum_auc = []
            for x_val, y_val in val_loader:
                # Again, sends data to same device as model
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                model.eval()
                # Makes predictions
                yhat = model(x_val.float()) # len=11, 一组batch的预测值

                # Computes validation loss
                loss = 0
                for i in range(tasks):
                    loss += loss_fn(yhat[:, i].float(), y_val[:, i].view(-1, 1).float())  * loss_weight[i]
                epoch_loss.append(loss.item())

        val_losses.append(np.mean(epoch_loss))

        epoch_len = len(str(n_epochs))
        mesg = f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' + \
                         f'train_loss: {losses[-1]:.5f} ' + \
                         f'valid_loss: {val_losses[-1]:.5f}'
        logging.info(mesg)

        # ************* Early Stopping ****************
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_losses[-1], model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # print(model.state_dict())
    print("loss: ", np.mean(losses))
    print("val_loss: ", np.mean(val_losses))

    
def predict():
    # 预测
    watch_pred = []
    share_pred = []
    with torch.no_grad():
        # Uses loader to fetch one mini-batch for testing
        for x_test in test_loader:
            # Again, sends data to same device as model
            x_test = x_test[0]
            x_test = x_test.to(device)

            model.eval()
            # Makes predictions
            yhat = model(x_test.float())
            yhat = yhat.squeeze(0)
            yhat = yhat.view(-1, 11)

            yhat_watch = yhat[:, :10]
            yhat_share = yhat[:, 10:]

            # save
            watch_pred.append(yhat_watch)
            share_pred.append(yhat_share)
    return watch_pred, share_pred

def save_submission(wt_preds, sh_preds):
    submission = pd.read_csv("./test.csv")
    submission['watch_label'] = wt_preds
    submission['is_share'] = sh_preds
    submission.save()
    

if __name__ == '__main__':
    
    # 加载数据
    dataset, watch_label, share_label = load_train_datas()
    # 将分类label 变为回归label
    watch_label_ratio = watch_label / 10
    # 划分数据集，得到训练集和验证集
    train_data, train_label, validation_data, validation_label = split_dataset(dataset, watch_label_ratio, share_label)
    
    
    # 定义模型
    model = MMOE(input_size=128, num_experts=8, experts_out=16, experts_hidden=16, towers_hidden=12, tasks=2)
    model = model.to(device)
    
    # 训练
    train()
    
    
    #加载预测数据集(test)
    test_data = load_test_datas()
    
    # 将测试数据组织为 DaTaloader
    test_loader = DataLoader(dataset=torch.utils.data.TensorDataset(torch.tensor(df_test.astype(float).to_numpy())), batch_size=BATCH_SIZE)
    
    wt_preds, sh_preds = predict()
    
    save_submission(wt_preds, sh_preds)