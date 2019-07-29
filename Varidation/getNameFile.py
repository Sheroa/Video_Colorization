import os
def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    numList = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            num_name = int(filespath.split('.')[0])
            numList.append(num_name)
        numList.sort()
        for i in range(len(numList)):
            ret.append(str(i)+'.jpg')
    return ret
def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

datasetL = get_jpgs('./judo')
text_save(datasetL, 'names.txt')
print(datasetL)
