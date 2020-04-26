from lib.metadata import Metadata

metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv')


print(metadata.classes())

#get metadata train set excluding first directory
training_set = metadata.train_set(1)
for cls in training_set:
    print(cls.name)
    print("  positive:")
    for sound in cls.positive[:2]: 
        #print first two of each class
        print("\t" + str(sound))
        #print(sound.load())
    print("  negative:")
    for sound in cls.negative[:2]: 
        #print first two of each class
        print("\t" + str(sound))
        #print(sound.load())

