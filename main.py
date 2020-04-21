from lib.metadata import Metadata

metadata = Metadata('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.load()

#just for testing purposes
metadata.print()

