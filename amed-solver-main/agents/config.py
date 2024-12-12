import inspect


def get_args()

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


class EDMCIFAR10Config(BaseConfig):
    seed = 1
    runner_class_name = 'DiffSampleOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        hidden_dim = 128,
        output_dim = 1,
        bottleneck_input_dim = 64,
        bottleneck_output_dim = 4,
        noise_channels = 8,
        embedding_type = 'positional',
        dataset_name = 'cifar10',
        img_resolution = 32,
        num_steps = 1000,
        # sampler_tea = None,
        # sampler_stu = None,
        M = None,
        guidance_type = None,
        guidance_rate = None,
        schedule_type = None,
        schedule_rho = None,
        afs = True,
        scale_dir = 0.01,
        scale_time = 0,
        max_order = None,
        predict_x0 = True,
        lower_order_final = True,

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'DiffSamplerPolicy'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt