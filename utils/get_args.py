import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--valid-freq", type=int, default=1, help="To valid every $valid-freq$ epochs.")

    parser.add_argument("--is-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-lr", type=float, default=0.00001, help="learning rate for CLIP")
    parser.add_argument("--resnet-lr", type=float, default=0.00001, help="learning rate for ResNet")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for all other modules")

    parser.add_argument("--detection-score", type=float, default=0.3)
    parser.add_argument("--bboxes-prune-pixel", type=int, default=15)
    parser.add_argument("--k-bits", type=int, default=32, help="length of hash codes.")
    parser.add_argument("--n-clusters", type=int, default=32, help="number of KMeans clusters.")
    parser.add_argument("--si-std", type=float, default=1.1, help="std threshold for SI")
    parser.add_argument("--truncation-factor", type=int, default=2, help="")

    # hyper-parameters
    # parser.add_argument("--hyper-lambda", type=float, default=0.99, help="proportionality constant when constructing B matrix.")

    parser.add_argument("--clip-path", type=str, default="./cache/ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--dataset", type=str, default="flickr25k", help="choose from [coco, flickr25k, nuswide]")
    parser.add_argument("--query-num", type=int, default=2000)
    parser.add_argument("--train-num", type=int, default=5000)
    
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--detection-dir", type=str, default="", help="path to directory containing images from object detection.")
    parser.add_argument("--result-name", type=str, default="result", help="result dir name.")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--bow-file", type=str, default="caption_one_hot.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")
    # parser.add_argument("--caption-one-hot-file", type=str, default="caption_one_hot.mat")

    parser.add_argument("--max-words", type=int, default=64, help="max number of words in a sequence")
    parser.add_argument("--resolution", type=int, default=224, help="images should be transformed to this resolution")

    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.05, help="Proportion of training to perform learning rate warmup.")

    args = parser.parse_args()

    import datetime
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_train:
        _time += "_test"
    k_bits = args.k_bits
    parser.add_argument("--save-dir", type=str, default=f"./{args.result_name}/{args.dataset}_{k_bits}/{_time}")
    args = parser.parse_args()

    return args