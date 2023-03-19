from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

def main():
    
    config = Cfg.load_config_from_name('vgg_seq2seq')

    print(config)
    trainer = Trainer(config)

    # if args.checkpoint:
    #     trainer.load_checkpoint(args.checkpoint)
        
    # trainer.train()

if __name__ == '__main__':
    main()
