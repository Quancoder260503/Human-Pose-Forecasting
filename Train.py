import numpy as np
from ConfigParser      import * 
from runner.RunnerUtil import *

def train(args):
    checkpoint = f"checkpoint/{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"
    runner = get_runner(args)(
        args = args, 
        pck_dict_fn      = get_ntu_pck_dict, 
        rmse_dict_fn     = get_ntu_mse_dict, 
        model_checkpoint = checkpoint,
    )
    device = args.device 
    print(device, 'is available')

    best_accuracy = 0
        
    loss_list, metric_list = [], []

    trainable_params = [p for p in runner.model.parameters() if p.requires_grad]
  
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_params:,}")


    print("Start Training ...\n")
    for epoch in range(runner.start_epoch, args.num_epochs):
        print(f"learning rate for epoch {epoch + 1}: {runner.lr_scheduler.get_last_lr()[0]}")
        runner.set_epoch(epoch)
        train_loss, train_predictions, train_rmse, val_loss, val_predictions, val_rmse = runner.process_per_epoch(epoch)

        train_acc = get_acc(train_predictions)
        val_acc   = get_acc(val_predictions)

        loss_list.append([train_loss, val_loss])
        metric_list.append([train_acc, val_acc])

        curr_train_total_acc, curr_val_total_acc = train_acc['total'], val_acc['total']

        print(f'Training | Loss : {train_loss:.6f} | PCK accuracy : {curr_train_total_acc:.4f} | PJRMSE : {train_rmse:.6f}')
        print(f'Validation | Loss : {val_loss:.6f} | PCK accuracy : {curr_val_total_acc:.4f}   | PJRMSE : {val_rmse:.6f}')

        if val_acc['total'] > best_accuracy : 
            best_accuracy = val_acc['total']
            save_checkpoint(
                epoch, 
                runner.model, 
                runner.optimizer, 
                runner.lr_scheduler, 
                checkpoint, 
            )

    plot(
        loss_list   = np.array(loss_list), 
        metric_list = np.array(metric_list), 
        save_path   = f'log/metrics/{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.png', 
        dataset     = args.dataset, 
    )

    # Saved the training logs 
    npy_data = dict()
    npy_data['loss_list']   = np.array(loss_list) 
    npy_data['metric_list'] = np.array(metric_list)
    log_saved_dir           = f'log/{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_pennaction_{args.num_training_frame}_{args.num_testing_frame}.npy'
    np.save(log_saved_dir, npy_data)



if __name__ == '__main__':
    args = parse_args()
    train(args)
