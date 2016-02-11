import os
import glob
import math
from docopt import docopt
import shutil
from random import shuffle

if __name__ == '__main__':

    pic_dir = '/Users/nikolaus/Documents/Stanford_Winter/CS231N/project231/Flickr'

    train_dir = '/Users/nikolaus/Documents/Stanford_Winter/CS231N/project231/torch_set/train'

    val_dir = '/Users/nikolaus/Documents/Stanford_Winter/CS231N/project231/torch_set/val'

    label_folders = [name for name in os.listdir(pic_dir)\
        if os.path.isdir(os.path.join(pic_dir, name))]

    print 'no categories',len(label_folders)
    #print label_folders
    for idx,label in enumerate(label_folders):
    	print 'progress ',label,' ',idx,'/',len(label_folders)
        path = pic_dir+'/'+label

        pic_list = glob.glob(os.path.join(path,'*.jpg'))
        shuffle(pic_list)
        #print pic_list[:20]
        #sys.exit(-1)
        train_len = int(0.8*len(pic_list))

        train_dir_label = train_dir +'/'+label
        val_dir_label = val_dir +'/'+label

        if not os.path.exists(train_dir_label):
            os.makedirs(train_dir_label)
    	if not os.path.exists(val_dir_label):
            os.makedirs(val_dir_label)

        for pic in pic_list[:train_len]:
            file_root, file_ext = os.path.splitext(pic)
            filename = os.path.basename(file_root)

            shutil.copy(pic,\
                os.path.join(train_dir_label, filename+file_ext))
            
        for pic in pic_list[train_len:]:
            file_root, file_ext = os.path.splitext(pic)
            filename = os.path.basename(file_root)

            shutil.copy(pic,\
                os.path.join(val_dir_label, filename+file_ext))
        #sys.exit(-1)
