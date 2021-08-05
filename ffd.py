import tarfile

tar = tarfile.open("EnglishImg.tgz", "r:gz")

tar.extractall()

tar.close()
