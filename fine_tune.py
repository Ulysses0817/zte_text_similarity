# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
print("fine_tuning model!")
import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu, K, search_layer
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense
set_gelu('tanh')  # 切换gelu版本

import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

random.seed(1996)
np.random.seed(1)
tf.set_random_seed(1)

import os, shutil

import argparse, ast
parser = argparse.ArgumentParser()
parser.description='Hi guys!'
parser.add_argument("-ml","--maxlen", help="序列最大长度,默认为128",type=int, default=128)
parser.add_argument("-e","--epochs", help="迭代次数,默认为30; Earlystopping 默认开启，patience为3，如欲修改，需要手动修改py文件。",type=int, default=30)
parser.add_argument("-b","--batch_size", help="训练批次大小,默认为64", type=int, default=64)
parser.add_argument("-cgp","--config_path", help="预训练模型配置文件路径,默认为./albert_config_small_google.json", \
                                            default='./albert_config_small_google.json')
parser.add_argument("-ckp","--checkpoint_path", help="预训练模型路径,默认为./model/albert/model.ckpt-250000", default='./model/albert/model.ckpt-250000')
parser.add_argument("-vp","--vocab_path", help="vocab文件路径,默认为./vocab.txt", default='./vocab.txt')
parser.add_argument("-lr","--learning_rate", help="学习率,默认为2e-5", default=2e-5, type=float)
parser.add_argument("-k","--kfold", help="k折交叉验证,默认为5",type=int, default=5)
parser.add_argument("-adver","--adver", help="是否启用对抗学习，默认为True", default=True, type=ast.literal_eval,)
parser.add_argument("-threshold","--threshold", help="是否启用对抗学习，默认为0.5", default=0.5, type=float)
parser.add_argument("-rp","--rank_predict", help="预测输出中是否令1与0的数量相等，注意，当该参数为True时，\
                                                    threshold参数会无效化.默认为True", default=True, type=ast.literal_eval,)

args = parser.parse_args()
tokenizer = Tokenizer(args.vocab_path, do_lower_case=True)

# %%
class data_generator(DataGenerator):
    """数据生成器
    """
    def set_random(self, mode="train", textrnd=False):
        self.textrnd = False
        if mode =="train":
            self.textrnd = textrnd
            self.random = True
        else:
            self.random = False
        print(mode, self.textrnd)

    def __iter__(self, x=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(self.random):
            if self.textrnd:
                if random.random()>0.5:
                    text1, text2 = text2, text1
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, max_length=args.maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def build_model(config_path, checkpoint_path):
    """ 
    加载预训练模型
    """
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='albert',# 也可以设置为 albert_unshared，收敛更快
        with_pool=True,
        return_keras_model=False,
    )

    output = Dropout(rate=0.3)(bert.model.output)
    output = Dense(
        units=2, activation='softmax', kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)

    return model

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数

def train(model, train_generator, valid_generator, lr=2e-5, freez=False, adver=False):
    """
    模型训练
    Args:
    lr -- 初始学习率
    freez -- 是否冻结bert预训练层
    """
    # reducelronplateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=verbose, mode='min', factor=0.2, patience=1)
    earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=3, restore_best_weights=True, mode='max')

    if freez:
        for layer in model.layers[2:-1]:
            print(layer.name, ":", layer.trainable)
            if layer.trainable: layer.trainable = False
        print(model.trainable_weights)
    else:
        for layer in model.layers[2:-1]:
            print(layer.name, ":", layer.trainable)
            if not layer.trainable: layer.trainable = True
        print(model.trainable_weights)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(lr),  # 用足够小的学习率  SGD(lr=0.0001, momentum=0.9)
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        metrics=['accuracy'],
    )

    # 写好函数后，启用对抗训练只需要一行代码
    if adver:
        print("activating adversarial training!")
        adversarial_training(model, 'Embedding-Token', 0.5)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        callbacks=[earlystopping],
        validation_data=valid_generator.forfit(),
        validation_steps=300,
        verbose=2
    )

    return model

def predict(model, data_generator):

    y_preds=[]
    for x_true, y_true in data_generator:
        y_preds.append(model.predict(x_true)[:, 1])

    return np.hstack(y_preds)

if __name__ == "__main__":
    # %%
    
    # 加载数据集
    train_data=pd.read_table(r'./data/train.txt',sep='\t', names=["text_a", "text_b", "label"])
    test_data=pd.read_table(r'./data/test.txt',sep='\t', names=["text_a", "text_b", "label"])
    print(test_data.shape, train_data.shape)
    # validation_steps = test_data.shape[0]//batch_size

    train_data = train_data.iloc[:100]
    test_data = test_data.iloc[:100]

    # 构造测试集迭代器
    test_generator = data_generator(test_data.values.tolist(), args.batch_size)
    test_generator.set_random(mode="test")

    reverse_text = test_data.values
    tmp = reverse_text[:, 0].copy()
    reverse_text[:, 0] = reverse_text[:, 1]
    reverse_text[:, 1] = tmp
    reverse_test_generator = data_generator(reverse_text.tolist(), args.batch_size)
    reverse_test_generator.set_random(mode="test")
    print(reverse_text.shape, reverse_text[0])

    # 训练主函数
    valid_datas = []
    skf = StratifiedKFold(n_splits=5, random_state=1996, shuffle=True)
    kf = KFold(n_splits=args.kfold, random_state=1996, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(kf.split(train_data)):#, train_data.label

        print("Fold:", fold)
        print(train_index.shape, valid_index.shape, train_data.loc[train_index, "label"].value_counts())
        
        # 构造数据迭代器
        train_generator = data_generator(train_data.loc[train_index].values.tolist(), args.batch_size)
        valid_generator = data_generator(train_data.loc[valid_index].values.tolist(), args.batch_size)
        train_generator.set_random(mode="train", textrnd=True)
        valid_generator.set_random(mode="val")
        
        # 加载预训练模型
        model = build_model(config_path=args.config_path, checkpoint_path=args.checkpoint_path)
        # model.summary()
        
        # 模型微调训练
        # model = train(model, train_generator, valid_generator, freez=True, lr=5e-5)
        model = train(model, train_generator, valid_generator, freez=False, lr=args.learning_rate, adver=True)
        
        # 模型保存
        model_name = f'best_model_{fold}.weights'
        model_path = './model/'+model_name
        model.save_weights(model_path)
        
        # 验证集预测，可用于stacking
        # valid_data = train_data.loc[valid_index].copy()
        # valid_data["pred"] = predict(model, valid_generator)
        # valid_datas.append(valid_data)
        
        # 测试集预测
        test_data[f"pred_{fold}"] = predict(model, test_generator)
        try:
            test_data[f"reverse_pred_{fold}"] = predict(model, reverse_test_generator)
        except Exception as e:
            print("reverse text failed...")
        test_data.to_csv("./results/test_probs.csv", index=False)
    
    # 生成提交文件
    csv = test_data.copy()
    if args.rank_predict:
        print("根据概率排序，令测试集label中1与0的个数相等")
        # 根据概率排序，令测试集label中1与0的个数相等
        csv["probs"] = csv.loc[:, [i for i in csv.columns if "pred" in i]].mean(1)
        idx1 = csv.sort_values("probs", ascending=False)[:6250].index
        idx0 = csv.sort_values("probs", ascending=False)[6250:].index
        csv.loc[idx1, "label"] = "1"
        csv.loc[idx0, "label"] = "0"
    else:
        print("将大于阈值的数据label置为1，此时阈值为：", args.threshold)
        # 根据阈值设置label，默认为0.5
        threshold = args.threshold
        csv["probs"] = csv.loc[:, [i for i in csv.columns if "pred" in i]].mean(1)>threshold
        csv["label"] = (1*csv["probs"]).astype("str")
    with open("./results/LiZeda_XJTU_predict.txt", "w") as f:
        f.write("\n".join(csv.label.values))
