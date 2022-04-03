import argparse
import sys
from models.use_models import UseModels


origin_path = sys.path
sys.path.append("..")
sys.path = origin_path


parser = argparse.ArgumentParser(description="Enter the parameters required for network operation")

parser.add_argument("--filePath",      type=str,      default="", required =True,
                    help="Address of the resource file")
parser.add_argument("--conditionPath", type=str,      default="", required =True,
                    help="Table address, corresponding to each data characteristic condition 0: delete the feature data 1: Continuous data  2: Discrete data")

parser.add_argument("--model_epochs",  type=int,      default=1,
                    help="Number of model runs default = 1")
parser.add_argument("--sim_size",      type=int,      default=15000,
                    help="Twin network training data size default = 15000")
parser.add_argument("--train_prob",    type=float,    default=0.002,
                    help="Proportion of reserved training data, default = 0.002")
parser.add_argument("--test_prob",     type=float,    default=0.003,
                    help="Proportion of retained test data, default = 0.003")
parser.add_argument("--test_len",      type=float,    default=0.25,
                    help="Division of original data and proportion of test data, default = 0.25")
parser.add_argument("--epochs",        type=int,      default=300,
                    help="Training times, default=300")
parser.add_argument("--batch_size",    type=int,      default=512,
                    help="Batch size, default=512")
parser.add_argument("--pdf",           type=bool,     default=True,
                    help="Save running picture data default=True")
parser.add_argument("--show",          type=bool,     default=True,
                    help="Whether to display running picture data default=True")

parser.add_argument("--epoc",          type=int,     default=1,
                    help="Suffix flag currently used to generate data default=1")

parser.add_argument("--CSDNN",         type=bool,     default=False,
                    help="Whether to run the CSDNN model default=False")
parser.add_argument("--SNN",           type=bool,     default=False,
                    help="Whether to run the SNN model default=False")
parser.add_argument("--FSSNN",         type=bool,     default=False,
                    help="Whether to run FSSNN model default=False")
parser.add_argument("--FSESNN",        type=bool,     default=True,
                    help="Whether to run the FSESNN model default=True")
args = parser.parse_args()
# print(args)

model = UseModels(sim_size=15000,           filePath='./data/feature_1.csv',
                  conditionPath='./data/feature_if.csv', train_prob=0.5,
                  test_prob=0.5,         test_len=0.25,
                  epochs=2,               batch_size=10,
                  pdf=True,                     show=False)

model.start(epoc=1)

