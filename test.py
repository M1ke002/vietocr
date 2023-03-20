from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

def main():
    
    config = Cfg.load_config_from_name('vgg_seq2seq')

    config['vocab'] = '!"$%&\'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_abcdefghijklmnopqrstuvwxyz{|}°²ÀÁÂÃÈÉÊÌÍÐÒÓÔÕÖÙÚÜÝàáâãèéêìíðòóôõöùúüýĀāĂăĐđĨĩŌōŨũŪūƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ–—’“”…− '

    dataset_params = {
        'name':'hw',
        'data_root':'dataset\\',
        'train_annotation':'Train_label.txt',
        'valid_annotation':'Test_label.txt'
    }

    params = {
            'print_every':20,
            'valid_every':50, #was 15*200
            'iters':400,
            'checkpoint':'\dataset\checkpoint\\transformerocr_checkpoint.pth',    
            'export':'\dataset\weights\\transformerocr.pth',
            'metrics': 200
            }

    dataloader_params = {'num_workers': 2, 'pin_memory': True}

    config['dataloader'].update(dataloader_params)
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'

    print(config)


    trainer = Trainer(config, pretrained=False)

    # if args.checkpoint:
    #     trainer.load_checkpoint(args.checkpoint)
        
    trainer.visualize_dataset()
    # trainer.train()

if __name__ == '__main__':
    main()

#conda activate vietocr