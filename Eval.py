from ConfigParser      import * 
from runner.RunnerUtil import *

def eval(args):
    checkpoint = f"checkpoint/{args.mode}/{args.joint_dim}D_{args.method_version}_{args.dataset}_{args.num_training_frame}_{args.num_testing_frame}.pth.tar"
    runner = NTURunner(
        args = args, 
        pck_dict_fn      = get_ntu_pck_dict, 
        rmse_dict_fn     = get_ntu_mse_dict, 
        model_checkpoint = checkpoint,
        eval             = True, 
    )
    device = args.device 
    print(device, 'is available')  
   # runner.evaluation()
    runner.inference()
   # runner.get_gradients()


if __name__ == '__main__':
    args = parse_args()
    eval(args) 


