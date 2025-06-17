# 173-Army-Classification

## Overview
<img src="/img/overview.png" width="75%" />

## 注意事项

### quick start
连接远程服务器：
```
ssh gpu4@58.20.54.203 -p 45220
```
运行下面的脚本开始训练模型：
```shell
python train_model.py
```
运行下面的脚本启动调度服务：
```shell
nohup python run_schedule.py > schedule.log &
```

### mBERT模型下载
+ 参考[Huggingface镜像站](https://hf-mirror.com/)或[教程链接](https://padeoe.com/huggingface-large-models-downloader/)；
+ 下载[mBERT](https://hf-mirror.com/bert-base-multilingual-uncased)所有的参数和配置文件；
+ 在项目根目录下创建`mbert`文件夹，并放入下载好的模型文件。

### 数据处理逻辑
+ 训练和测试数据转换和生成：
  + 首先执行`/preprocess/preprocess.py`处理带有标号的数据；
  + 然后执行`/preprocess/split_data.py`划分标号数据为训练数据和测试数据：
  ```
  python split_data.py 0.2  // 0.2为传递的test_ratio参数，用于设置测试集的大小
  ```
+ 预测数据处理：
  + 执行`/preprocess/preprocess_with_arg.py`处理待预测数据。

### 项目复盘
+ TAG中每个节点的文本特征由mbert预先处理成特征向量，训练时将特征矩阵加载入memory bank中进行查询使用 
+ @torch.no_grad()函数装饰器仅替代with torch.no_grad():语法，仍然需要设置model.eval()才能保证预测结果一致
+ 保存模型参数有两种形式1.保存整个模型以及2.仅保存模型参数state dict，理解上两者区别不是很大，磁盘空间占用差距不明显
+ 保存项目中的依赖文件:
```
pipreqs ./ --encoding=utf8 --force /* --force表示重写requirement.txt文件 */
```
+ 切换解释器：ctrl+shift+p打开命令栏 -> Python: Select Interpreter -> 选择目标conda环境
+ 日志依赖包选择的是loguru
+ tensorboard可视化工具，[教程链接](https://pytorch.org/docs/stable/tensorboard.html)
```python
writer = SummaryWriter(log_dir=tensorboard_log_dir)
writer.add_graph(model, input_to_model)
writer.add_scalar(tag, scalar_value, global_step=None) # 此外，还可以在同一坐标系中添加多个scalars
```
```shell
tensorboard --logdir=log/tensorboard --port 8123
```
+ github遇到"ssh: connect to host github.com port 22: Connection refused"：[解决办法：使用443端口连接github](https://zhuanlan.zhihu.com/p/521340971)