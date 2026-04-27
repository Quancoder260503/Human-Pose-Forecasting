from ConfigParser import parse_args
from runner.RunnerUtil import * 


def eval(args):
    checkpoint = f"{args.mode}/{args.model_input_type}_{args.model}_{args.model_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"
    runner = get_runner(args)(
        args = args, 
        pck_dict_fn      = get_ntu_pck_dict, 
        rmse_dict_fn     = get_ntu_mse_dict, 
        model_checkpoint = checkpoint,
    )
    device = args.device 
    print(device, 'is available')  
    runner.inference()


if __name__ == '__main__':
    args = parse_args()
    eval(args) 


