import os
names = os.listdir('img')  #要依据此路径下的文件名做删除操作
train_val = []
for name in names:
    index = name.rfind('.')
    name = name[:index]
    train_val.append(name+'.jpg')#其他图片格式修改后缀 eg:'.png'

delet=os.listdir('img - 副本')#被清洗的文件路径
for file in delet:
    if  (file in train_val):
        del_file = 'img - 副本'+'\\' + file #当代码和要删除的文件不在同一个文件夹时，必须使用绝对路径
        os.remove(del_file)#删除文件
        print("已经删除：",del_file)
