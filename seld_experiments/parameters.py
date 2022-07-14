# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        fib_tes=False,
        fib_N = 256,

        base_tess_dist = 5,
        base_tess_mode = False,

        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,
        # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/sk8974/experiments/dcase22/data/dcase2022/comb/',
        unet_norm_feat_dir='/scratch/sk8974/experiments/dcase22/unet/eigen_trained_unet/data/norm_in_2022/',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/sk8974/experiments/dcase22/data/processed/comb_base_feat/',

        model_dir='models/',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='mic',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,  # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,  # Feature sequence length
        batch_size=128,  # Batch size
        dropout_rate=0.05,  # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,  # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],
        # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,  # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,  # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',  # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,
        small_batch=False,
        patience=20,  # Stop training if patience is reached
        unique_classes = 13,
        use_comb_loss = False,
        loss_thr_scale = 1e-3,
    )

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling


    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '8':
        print("MIC + SALSA + ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False
        params['nb_cnn2d_filt'] = 128
        params['nb_rnn_layers'] = 2
    elif argv == '9':
        print("MIC + SALSA + ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False
        params['nb_cnn2d_filt'] = 128
        params['nb_rnn_layers'] = 3

    elif argv == '10':
        print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt'] = 128
        params['nb_rnn_layers'] = 2
    elif argv == '11':
        print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt'] = 128
        params['nb_rnn_layers'] = 2
    elif argv == '12':
        print("MIC + SALSA + multi ACCDOA\n")
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True
        params['nb_cnn2d_filt'] = 256
        params['nb_rnn_layers'] = 2
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    elif argv == '13':
        print("UNET + multi ACCDOA + testing\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = True
        params['batch_size'] = 32
        #         params['lr']=3e-4
        params['dropout_rate'] = 0.0
        params['patience'] = 1000

    elif argv == '14':
        print("UNET + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = False
        params['batch_size'] = 32
#         params['lr'] = 3e-4
        params['patience'] = 20

    elif argv == '15':
        print("DCASE + UNET + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = False
        params['batch_size'] = 32
        # params['lr'] = 1e-2
        params['patience'] = 20
        params['t_pool_size'] = [12, 1, 1]
        params['small_batch'] = False
        params['patience'] = 10
        params['unet_norm_feat_dir'] = '/scratch/sk8974/experiments/dcase22/unet/dcase_trained_unet/data/norm_in_2022/'
        params['t_unet_pool_size'] = [12, 1, 1]
        params['t_base_pool_size'] = [5, 1, 1]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/models/15_1_dev_split0_multiaccdoa_mic_gcc_model.h5'

    elif argv == '16':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_feat/'
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb/'
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/models/16_1_dev_split0_multiaccdoa_mic_gcc_model.h5'

    elif argv == '17':
        print("DCASE + UNET + multi ACCDOA+ comb\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = False
        params['batch_size'] = 32
        # params['lr'] = 1e-2
        params['t_pool_size'] = [12, 1, 1]
        params['small_batch'] = False
        params['patience'] = 10
        params['unet_norm_feat_dir'] = '/scratch/sk8974/experiments/dcase22/unet/dcase_trained_unet/data_correct/norm_in_2022/'
        params['t_unet_pool_size'] = [12, 1, 1]
        params['t_base_pool_size'] = [5, 1, 1]
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_feat/'
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb/'
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/models/17_1_dev_split0_multiaccdoa_mic_gcc_model.h5'

    elif argv == '18':
        print("DCASE + UNET + multi ACCDOA+ comb + batch 128\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = False
        params['batch_size'] = 64
        # params['lr'] = 1e-2
        params['t_pool_size'] = [12, 1, 1]
        params['small_batch'] = False
        params['patience'] = 20
        params[
            'unet_norm_feat_dir'] = '/scratch/sk8974/experiments/dcase22/unet/dcase_trained_unet/data_ablated/norm_in_2022/'
        params['t_unet_pool_size'] = [12, 1, 1]
        params['t_base_pool_size'] = [5, 1, 1]
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_feat/'
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb/'


    elif argv == '19':
        print("UNET + multi ACCDOA+ comb + batch 128\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['small_batch'] = False
        params['batch_size'] = 64
        # params['lr'] = 1e-2
        params['t_pool_size'] = [12, 1, 1]
        params['small_batch'] = False
        params['patience'] = 20
        params[
            'unet_norm_feat_dir'] = '/scratch/sk8974/experiments/dcase22/unet/dcase_trained_unet/data_ablated/norm_in_2022/'
        params['t_unet_pool_size'] = [12, 1, 1]
        params['t_base_pool_size'] = [5, 1, 1]
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_feat/'
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb/'

    elif argv == '20':
        print("BASE + multi ACC + comb + fib_tes\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir']='/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib/'
        params['fib_N'] = 256

    elif argv == '21':
        print("BASE + multi ACC + comb + fib_tes + 65536\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir']='/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_65536/'
        params['fib_N'] = 65536

    elif argv == '22':
        print("BASE + multi ACC + comb + fib_tes + 131072\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir']='/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_131072/'
        params['fib_N'] = 131072

    elif argv == '23':
        print("BASE + multi ACC + new synth + fib_tes + 32768\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_32768_1/'
        params['fib_N'] = 32768

    elif argv == '24':
        print("BASE + multi ACC + comb + fib_tes + 128\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_128/'
        params['fib_N'] = 128

    elif argv == '25':
        print("BASE + multi ACC + comb + fib_tes + 256\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_256/'
        params['fib_N'] = 256

    elif argv == '26':
        print("BASE + multi ACC + synth + fib_tes + 512\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_512_1/'
        params['fib_N'] = 512

    elif argv == '27':
        print("BASE + multi ACC + comb + fib_tes + 32\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32/'
        params['fib_N'] = 32

    elif argv == '28':
        print("BASE + multi ACC + comb + fib_tes + 64\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_64/'
        params['fib_N'] = 64

    elif argv == '29':
        print("BASE + multi ACC + comb + fib_tes + 2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_2/'
        params['fib_N'] = 2

    elif argv == '30':
        print("BASE + multi ACC + comb + fib_tes + 4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_4/'
        params['fib_N'] = 4

    elif argv == '31':
        print("BASE + multi ACC + comb + fib_tes + 8\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_8/'
        params['fib_N'] = 8

    elif argv == '32':
        print("BASE + multi ACC + comb + fib_tes + 16\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_16/'
        params['fib_N'] = 16

    elif argv == '33':
        print("BASE + multi ACC + created_data \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_1/'

    elif argv == '34':
        print("BASE + multi ACC + created_data \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_1_new/'

    elif argv == '35':
        print("BASE + multi ACC + new synth + fib_tes + 16\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_16_1_new/'
        params['fib_N'] = 16

    elif argv == '36':
        print("BASE + multi ACC + new synth + fib_tes + 128\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_128_1_new/'
        params['fib_N'] = 128

    elif argv == '37':
        print("BASE + multi ACC + new synth + fib_tes + 1024\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_1024_1_new/'
        params['fib_N'] = 1024

    elif argv == '38':
        print("BASE + multi ACC + new synth + fib_tes + 32768\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_32768_1_new/'
        params['fib_N'] = 32768

    elif argv == '39':
        print("BASE + multi ACC + new synth + fib_tes + 65536\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['fib_tes'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_65536_1_new/'
        params['fib_N'] = 65536

    elif argv == '40':
        print("BASE + multi ACC + comb_data + threshold 5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 5

    elif argv == '41':
        print("BASE + multi ACC + comb_data + threshold 10 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 10

    elif argv == '42':
        print("BASE + multi ACC + comb_data + threshold 40 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 40

    elif argv == '43':
        print("BASE + multi ACC + comb_data + threshold 80 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 80

    elif argv == '44':
        print("BASE + multi ACC + comb_data + threshold 20 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20

    elif argv == '45':
        print("BASE + multi ACC + comb_data + tess 8 + threshold 67 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 67
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_8/'
        params['fib_N'] = 8
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/45_1_dev_split0_multiaccdoa_mic_gcc_20220625170018_model.h5'


    elif argv == '46':
        print("BASE + multi ACC + comb_data + tess 32 + threshold 35 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 35
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32/'
        params['fib_N'] = 32
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/46_1_dev_split0_multiaccdoa_mic_gcc_20220625165824_model.h5'

    elif argv == '47':
        print("BASE + multi ACC + comb_data + tess 512 + threshold 9 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 9
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_512/'
        params['fib_N'] = 512

    elif argv == '48':
        print("BASE + multi ACC + comb_data + tess 2048 + threshold 5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 5
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_2048/'
        params['fib_N'] = 2048

    elif argv == '49':
        print("BASE + multi ACC + comb_data + tess 100 + threshold 20 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_100/'
        params['fib_N'] = 100
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/49_1_dev_split0_multiaccdoa_mic_gcc_20220625204214_model.h5'

    elif argv == '50':
        print("BASE + multi ACC + comb_data + base_tess 5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_mode'] = True
        params['base_tess_dist'] = 5
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_base_tess_5/'
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/50_1_dev_split0_multiaccdoa_mic_gcc_20220627233333_model.h5'

    elif argv == '51':
        print("BASE + multi ACC + comb_data + base_tess 10 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_mode'] = True
        params['base_tess_dist'] = 10
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_base_tess_10/'
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/51_1_dev_split0_multiaccdoa_mic_gcc_20220628034639_model.h5'

    elif argv == '52':
        print("BASE + multi ACC + comb_data + base_tess 45 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_mode'] = True
        params['base_tess_dist'] = 45
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_base_tess_45/'

    elif argv == '53':
        print("BASE + multi ACC + new synth + base_tess + 10\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_dist'] = 10
        params['base_tess_mode'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_tess_10/'
        params['unique_classes'] = 1

    elif argv == '54':
        print("BASE + multi ACC + new synth + base_tess + 20\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_dist'] = 20
        params['base_tess_mode'] = True
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_tess_20/'
        params['unique_classes'] = 1

    elif argv == '55':
        print("MIC + GCC + multi ACCDOA comb\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True

    elif argv == '56':
        print("MIC + GCC + multi ACCDOA comb + base thr\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 0.0

    elif argv == '57':
        print("MIC + GCC + multi ACCDOA comb + base thr 2e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '58':
        print("MIC + GCC + multi ACCDOA comb + base thr 2e-6\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-6
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/58_1_dev_split0_multiaccdoa_mic_gcc_20220701231759_model.h5'

    elif argv == '59':
        print("BASE + multi ACC + comb_data + fib tess 8 + threshold 67 + new loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 67
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_8/'
        params['fib_N'] = 8
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '60':
        print("BASE + multi ACC + comb_data + fib tess 32 + threshold 35 + new loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 35
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32/'
        params['fib_N'] = 32
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '61':
        print("BASE + multi ACC + comb_data + fib tess 100 + threshold 20 + new loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_100/'
        params['fib_N'] = 100
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '62':
        print("BASE + multi ACC + comb_data + fib tess 512 + threshold 9 + new loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 9
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_512/'
        params['fib_N'] = 512
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '63':
        print("BASE + multi ACC + comb_data + fib tess 2048 + threshold 5 + new loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 5
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_2048/'
        params['fib_N'] = 2048
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5

    elif argv == '64':
        print("MIC + GCC + multi ACCDOA + combdata + base tess + newloss wt. 2e-6 + thr 15 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-6
        params['lad_doa_thresh'] = 15

    elif argv == '65':
        print("MIC + GCC + multi ACCDOA + combdata + base tess + newloss wt. 2e-5 + thr 15 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5
        params['lad_doa_thresh'] = 15

    elif argv == '66':
        print("Base + fake debug \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/fake_data_debug/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/fake_data_debug/'

    elif argv == '67':
        print("MIC + GCC + multi ACCDOA + combdata + base tess + updnewloss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '68':
        print("MIC + GCC + multi ACCDOA + combdata + base tess + updnewloss wt 1\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '69':
        print("MIC + GCC + multi ACCDOA + combdata + base tess + updnewloss wt 2e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '70':
        print("MIC + GCC + multi ACCDOA + dummy \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '71':
        print("MIC + GCC + multi ACCDOA + finetune + new loss + thr 20 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '72':
        print("MIC + GCC + multi ACCDOA + finetune + new loss + thr 10 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 10
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '73':
        print("MIC + GCC + multi ACCDOA + finetune + new loss + thr 0 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 0
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/6_1_dev_split0_multiaccdoa_mic_gcc_20220609220551_model.h5'

    elif argv == '74':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20

    elif argv == '75':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 10 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 10

    elif argv == '76':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 0 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 0

    elif argv == '77':
        print("BASE + multi ACC + comb_data + fib tess 512 + threshold 9 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 9
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_512/'
        params['fib_N'] = 512
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3

    elif argv == '78':
        print("BASE + multi ACC + comb_data + tess 100 + threshold 20 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['loss_thr_scale'] = 1e-3
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_100/'
        params['fib_N'] = 100

    elif argv == '79':
        print("BASE + multi ACC + comb_data + tess 32k + threshold 1 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 1
        params['fib_tes'] = True
        params['loss_thr_scale'] = 1e-3
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['fib_N'] = 32768

    elif argv == '80':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 + wt 2e-4 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-4
        params['lad_doa_thresh'] = 20

    elif argv == '81':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 + wt 2e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-5
        params['lad_doa_thresh'] = 20

    elif argv == '82':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 + wt 2e-6 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-6
        params['lad_doa_thresh'] = 20

    elif argv == '83':
        print("BASE + multi ACC + comb_data + threshold 5 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 5

    elif argv == '84':
        print("BASE + multi ACC + comb_data + threshold 10 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 10

    elif argv == '85':
        print("BASE + multi ACC + comb_data + threshold 20 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20

    elif argv == '86':
        print("BASE + multi ACC + comb_data + threshold 40 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 40

    elif argv == '87':
        print("BASE + multi ACC + comb_data + threshold 80 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 80

    elif argv == '88':
        print("BASE + multi ACC + comb_data + base_down_sample 5 + old loss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_mode'] = True
        params['base_tess_dist'] = 5
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_base_tess_5/'

    elif argv == '89':
        print("BASE + multi ACC + comb_data + base_down_sample 10 + oldloss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['base_tess_mode'] = True
        params['base_tess_dist'] = 10
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_base_tess_10/'

    elif argv == '90':
        print("BASE + multi ACC + comb_data + tess 8 + threshold 67 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 67
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_8/'
        params['fib_N'] = 8


    elif argv == '91':
        print("BASE + multi ACC + comb_data + tess 32 + threshold 35 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 35
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32/'
        params['fib_N'] = 32

    elif argv == '92':
        print("BASE + multi ACC + comb_data + tess 100 + threshold 20 + old loss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_100/'
        params['fib_N'] = 100

    elif argv == '93':
        print("BASE + multi ACC + comb_data + tess 512 + threshold 9 + old loss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 9
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_512/'
        params['fib_N'] = 512

    elif argv == '94':
        print("BASE + multi ACC + comb_data + tess 2048 + threshold 5 + old loss \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 5
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_2048/'
        params['fib_N'] = 2048

    elif argv == '95':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 + wt 1e-2 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-2
        params['lad_doa_thresh'] = 20

    elif argv == '96':
        print("MIC + GCC + multi ACCDOA + scratch + new loss + thr 20 + wt 1e-1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-1
        params['lad_doa_thresh'] = 20

    elif argv == '97':
        print("BASE + multi ACC + comb_data + fib tess 512 + threshold 9 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 9
        params['fib_tes'] = True
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_512/'
        params['fib_N'] = 512
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['use_comb_loss'] = True

    elif argv == '98':
        print("BASE + multi ACC + comb_data + tess 100 + threshold 20 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['loss_thr_scale'] = 1e-3
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_100/'
        params['fib_N'] = 100
        params['use_comb_loss'] = True

    elif argv == '99':
        print("BASE + multi ACC + comb_data + tess 32k + threshold 1 + new loss scratch \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['lad_doa_thresh'] = 1
        params['fib_tes'] = True
        params['loss_thr_scale'] = 1e-3
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['fib_N'] = 32768
        params['use_comb_loss'] = True


    elif argv == '100':
        print("BASE + multi ACC + iran_data + new_loss + thr 1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_1_new/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 1
        params['unique_classes'] = 1

    elif argv == '101':
        print("BASE + multi ACC + iran_data + new_loss + thr 0.75 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_base_1_new/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 0.75
        params['unique_classes'] = 1

    elif argv == '102':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k + correct_eval \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '103':
        print("BASE + multi ACC + comb_data + old_loss + fib_tess 32k + correct_eval \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '105':
        print("BASE + multi ACC + iran_data + new_loss + thr 20 + fib_tess 32k + correct_eval \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/synthetic_dcase_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/synth_32768_1_new/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768
        params['unique_classes'] = 1

    elif argv == '106':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k + correct_eval + wt 0 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 0
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '107':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k + correct_eval + wt 2e-4 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-4
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '108':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k + correct_eval + wt 0 + not float\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 0
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '109':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k + correct_eval + wt 2e-4 + not float\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/comb_fibtess_32k/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_fib_32k/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-4
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '110':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_salsalite'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa/'

    elif argv == '112':
        print("FOA + multi ACCDOA + scratch + new loss +  wt 1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['use_salsalite'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20

    elif argv == '113':
        print("BASE + multi ACC + comb_data + old_loss + thr 20 + fib_tess 32k\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa_fib_32k/'
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768

    elif argv == '114':
        print("BASE + multi ACC + comb_data + new_loss + thr 20 + fib_tess 32k\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True
        params['finetune_mode'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa_fib_32k/'
        params['lad_doa_thresh'] = 20
        params['fib_tes'] = True
        params['fib_N'] = 32768
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3

    elif argv == '115':
        print("FOA + multi ACCDOA + fine tune + scratch + new loss +  wt 1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-3
        params['lad_doa_thresh'] = 20
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/110_1_dev_split0_multiaccdoa_foa_20220713191105_model.h5'

    elif argv == '116':
        print("FOA + multi ACCDOA + fine tune + scratch + new loss +  wt 2e-4 \n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 2e-4
        params['lad_doa_thresh'] = 20
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/110_1_dev_split0_multiaccdoa_foa_20220713191105_model.h5'

    elif argv == '117':
        print("FOA + multi ACCDOA + fine tune + scratch + new loss +  wt 1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['use_salsalite'] = False
        params['dataset_dir'] = '/scratch/sk8974/experiments/dcase22/data/dcase2022/foa/comb/'
        params['feat_label_dir'] = '/scratch/sk8974/experiments/dcase22/data/processed/comb_base_foa/'
        params['use_comb_loss'] = True
        params['loss_thr_scale'] = 1e-5
        params['lad_doa_thresh'] = 20
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/scratch/sk8974/experiments/dcase22/run/models/110_1_dev_split0_multiaccdoa_foa_20220713191105_model.h5'

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    #
    # if '2020' in params['dataset_dir']:
    #     params['unique_classes'] = 14
    # elif '2021' in params['dataset_dir']:
    #     params['unique_classes'] = 12
    # elif '2022' in params['dataset_dir']:
    #     params['unique_classes'] = 1
    # else:
    #     params['unique_classes'] = 1

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
