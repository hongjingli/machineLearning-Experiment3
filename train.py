from feature import NPDFeature
import os
from ensemble import AdaBoostClassifier
from PIL import Image
import  numpy
import pickle
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import tree

#定义资源路径
path_face_origin = 'datasets/original/face/'
path_nonface_origin = 'datasets/original/nonface/'
path_face_purpose = 'datasets/purpose/face/'
path_nonface_purpose = 'datasets/purpose/nonface/'

#图片处理
def process_image(path, purpose_path):
    files = os.listdir(path)
    for i in files:
        #将图片转化为灰度图
        im = Image.open(path + i).convert('L')
        #将图片转化为24*24的大小
        im.resize((24, 24),Image.ANTIALIAS)
        #保存图片
        im.save(purpose_path + i)

#获取特征
def get_feature(path):
    features = numpy.array([])
    files = os.listdir(path)
    for k in range(len(files)):
        im = Image.open(path + files[k])
        image = numpy.ones(shape=(24, 24), dtype=int)
        for i in range(24):
            for j in range(24):
                image[i][j] = im.getpixel((i, j))
        NPDFeature1 = NPDFeature(image)
        feature = NPDFeature1.extract()
        features = numpy.concatenate((features, feature))
    return features

#预处理过程
def pre_process():
    face_features = numpy.array([])
    nonface_features = numpy.array([])
    features = numpy.array([])

    #转化图片
    process_image(path_face_origin, path_face_purpose)
    process_image(path_nonface_origin, path_nonface_purpose)
    
    num_face = len(os.listdir('datasets/original/face'))
    num_nonface = len(os.listdir('datasets/original/nonface'))

    #获取特征
    face_features = get_feature(path_face_purpose)
    nonface_features = get_feature(path_nonface_purpose)

    print(face_features.shape,nonface_features.shape)
    print(num_face,num_nonface)
    
    #改变特征形状
    face_features.shape = num_face, 165600 
    nonface_features.shape = num_nonface, 165600

    #准备加入label
    face_y = numpy.ones(shape=(num_face, 1))
    nonface_y = numpy.zeros(shape=(num_nonface, 1))
    nonface_y -= numpy.ones(shape=(num_nonface, 1))

    #加入label
    face_features = numpy.concatenate([face_features, face_y], axis=1)
    nonface_features = numpy.concatenate([nonface_features, nonface_y], axis=1)

    #将所有数据合并
    features = numpy.row_stack((face_features,nonface_features))

    #写入缓存
    AdaBoostClassifier.save(features,"feature.data")
    return features

if __name__ == "__main__":
    #Data格式为[X:y]
    if os.path.exists('feature.data'): #如果预处理过，直接用load()读取数据
        Data = AdaBoostClassifier.load('feature.data')
    else :
        Data = pre_process()

    #将X_data与y_data分开
    X_data,y_data = Data[:,:-1],Data[:,-1]

    #切分训练集与验证集
    X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.3,random_state=10)

    print(len(y_train),len(y_test))

    #进行AdaBoost训练
    mode = tree.DecisionTreeClassifier(max_depth=1)
    adaboost=AdaBoostClassifier(mode,20)
    adaboost.fit(X_train,y_train)

    #得到预测结果
    y_predict=adaboost.predict(X_test)

    #输出正确率
    count=0
    for i in range(len(y_test)):
        if y_test[i]==y_predict[i]:
            count=count+1
    target_names = ['1', '-1']
    print(count/len(y_test))

    #调用classification_report获得预测结果
    report=classification_report(y_test, y_predict, target_names=target_names)

    #写入report.txt
    with open("report.txt", 'w') as f:
        f.write(report)
    print(report)
    

