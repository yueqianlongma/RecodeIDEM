import argparse
import sys
from models.use_models import UseModels


origin_path = sys.path
sys.path.append("..")
sys.path = origin_path


parser = argparse.ArgumentParser(description="输入网络运行所需的参数")

parser.add_argument("--filePath",      type=str,      default="", required =True,
                    help="资源文件的地址")
parser.add_argument("--conditionPath", type=str,      default="", required =True,
                    help="表地址，对应每个数据特征条件注明 0：删除该特征数据 1：连续数据  2：离散数据")

parser.add_argument("--model_epochs",  type=int,      default=1,
                    help="模型运行次数 default = 1")
parser.add_argument("--sim_size",      type=int,      default=15000,
                    help="孪生网络训练数据大小 default = 15000")
parser.add_argument("--train_prob",    type=float,    default=0.002,
                    help="保留使用的训练数据占比, default = 0.002")
parser.add_argument("--test_prob",     type=float,    default=0.003,
                    help="保留使用的测试数据占比, default = 0.003")
parser.add_argument("--test_len",      type=float,    default=0.25,
                    help="原始数据划分，测试数据所占比率, default = 0.25")
parser.add_argument("--epochs",        type=int,      default=300,
                    help="训练次数, default=300")
parser.add_argument("--batch_size",    type=int,      default=512,
                    help="批处理大小, default=512")
parser.add_argument("--pdf",           type=bool,     default=True,
                    help="是否保存运行图片数据 default=True")
parser.add_argument("--show",          type=bool,     default=True,
                    help="是否显示运行图片数据 default=True")

parser.add_argument("--epoc",          type=int,     default=1,
                    help="目前用于生成数据的后缀标志 default=1")

parser.add_argument("--CSDNN",         type=bool,     default=False,
                    help="是否运行CSDNN模型 default=False")
parser.add_argument("--SNN",           type=bool,     default=False,
                    help="是否运行SNN模型 default=False")
parser.add_argument("--FSSNN",         type=bool,     default=False,
                    help="是否运行FSSNN模型 default=False")
parser.add_argument("--FSESNN",        type=bool,     default=True,
                    help="是否运行FSESNN模型 default=True")
args = parser.parse_args()
# print(args)

model = UseModels(sim_size=15000,           filePath='./data/feature_1.csv',
                  conditionPath='./data/feature_if.csv', train_prob=0.5,
                  test_prob=0.5,         test_len=0.25,
                  epochs=2,               batch_size=10,
                  pdf=True,                     show=False)

model.start(epoc=1)

