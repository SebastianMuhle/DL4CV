from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from itertools import chain, repeat, cycle
import numpy as np

#Generator
class MultilabelGenerator:
    def __init__(self,path_to_data, idg, photo_name_to_label_dict, batch_size=256, target_size=(32,32), train_or_valid='train'):
        self.directory_generator = idg.flow_from_directory(path_to_data, batch_size=batch_size, target_size=target_size, classes=[train_or_valid], shuffle=False)
        self.photo_name_to_label_dict = photo_name_to_label_dict
        self.batch_size = batch_size
        self.target_size = target_size
        self.train_or_valid = train_or_valid

    def flow(self):
        names_generator = self.grouper(self.batch_size, self.directory_generator.filenames)
        for (X_batch, _), names in zip(self.directory_generator, names_generator):
            names = [n.split('/')[-1].replace('.jpg','') for n in names]
            if (self.train_or_valid == 'test'):
                yield X_batch
            else:
                targets = [self.photo_name_to_label_dict[int(x)] for x in names]
                yield X_batch, targets

    def grouper(self, n, iterable, padvalue=None):
        g = cycle(zip(*[chain(iterable, repeat(padvalue, n-1))]*n))
        for batch in g:
            yield list(filter(None, batch))