import sys, time, os, argparse, warnings, glob, torch
from tools import *
from model import *
from dataLoader import *

# Training settings
parser = argparse.ArgumentParser(description = "Stage I, self-supervsied speaker recognition with contrastive learning.")
parser.add_argument('--max_frames',        type=int,   default=180,          help='Input length to the network, 1.8s')
parser.add_argument('--batch_size',        type=int,   default=300,          help='Batch size, bigger is better')
parser.add_argument('--n_cpu',             type=int,   default=4,            help='Number of loader threads')
parser.add_argument('--test_interval',     type=int,   default=1,            help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',         type=int,   default=80,           help='Maximum number of epochs')
parser.add_argument('--lr',                type=float, default=0.001,        help='Learning rate')
parser.add_argument("--lr_decay",          type=float, default=0.95,         help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--initial_model',     type=str,   default="",           help='Initial model path')
parser.add_argument('--save_path',         type=str,   default="./checkpoint",           help='Path for model and scores.txt')
parser.add_argument('--save_path_kaggle',  type=str,   default="",           help='Path for model and scores.txt in kaggle')
parser.add_argument('--train_list',   type=str, default="/kaggle/input/list-file/list_file/train.txt",help='Path for Vox2 list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--val_list',     type=str, default="/kaggle/input/list-file/list_file/valid.txt", help='Path for Vox_O list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--train_path',   type=str, default="/kaggle/input/data-cluster/data_cluster/data_set", help='Path to the Vox2 set')
parser.add_argument('--val_path',     type=str, default="/kaggle/input/data-cluster/data_cluster/data_set", help='Path to the Vox_O set')
parser.add_argument('--musan_path',   type=str, default="/kaggle/input/musan-noise", help='Path to the musan set')
parser.add_argument('--data_path',   type=str, default="/kaggle/input/data-cluster/data_cluster/data_set")
parser.add_argument('--excel_file',   type=str, default="")
parser.add_argument('--id',   type=str, default="")
parser.add_argument('--saved_file_cluster',   type=str, default="")
parser.add_argument('--model_cluster',   type=str, default="")
parser.add_argument('--K',   type=str, default="")
parser.add_argument('--min_samples',   type=str, default="")
parser.add_argument('--xi',   type=str, default="")
parser.add_argument('--min_cluster_size',   type=str, default="")
parser.add_argument('--max_eps2',   type=str, default="")
parser.add_argument('--n_jobs2',   type=str, default="")
parser.add_argument('--random_state',   type=str, default="")
parser.add_argument('--max_iter',   type=str, default="")
parser.add_argument('--n_init',   type=str, default="")
parser.add_argument('--metric',   type=str, default="")
parser.add_argument('--eval',              dest='eval', action='store_true', help='Do evaluation only')
args = parser.parse_args()
# Initialization  
model_save_path     = args.save_path+"/model"
print("model_save_path: ", model_save_path)
result_save_path    = args.save_path+"/result"
saved_path_kaggle = args.save_path_kaggle+"/model"
print("saved_path_kaggle: ", saved_path_kaggle)


Trainer = model(**vars(args)) # Define the framework
modelfiles = glob.glob('%s/model0*.model'%model_save_path) # Search the existed model files
modelfiles.sort()

print("load check point: ", modelfiles[-1])
Trainer.load_network(modelfiles[-1])
if args.model_cluster == "OPTICS":
    Trainer.clustering_OPTICS(args.id, args.val_path, args.saved_file_cluster, args.min_samples, args.min_cluster_size, args.max_eps2, args.n_jobs2, args.metric)
if args.model_cluster == "KMEANS":
    Trainer.clustering_KMEANS(args.id, args.val_path, args.saved_file_cluster, args.K, args.random_state, args.max_iter, args.n_init)
if args.model_cluster == "DBSCAN":
    Trainer.clustering_DBSCAN(args.id, args.val_path, args.saved_file_cluster, args.min_samples, args.max_eps2, args.n_jobs2, args.metric)
if args.model_cluster == "All_id":
    Trainer.clustering_All(args.data_path, args.excel_file, args.min_samples, args.min_cluster_size, args.max_eps2, args.n_jobs2, args.metric)
