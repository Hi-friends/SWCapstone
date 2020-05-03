import numpy as np
import PIL
from PIL import Image

celebA_path = "D:/face_db/CelebA"
output_path = "D:/face_db/SHN/data"

part_dict = {}
with open("{0}/list_eval_partition.txt".format(celebA_path)) as file:
	for line in file:
		token = line.strip().split()
		part_dict[token[0]] = token[1]

for type in ["train","test"]:
        L = []
        for key in part_dict:
                if part_dict[key]==("0" if type=="train" else "2" if type=="test" else None):
                        L.append(key)
        count = len(L)
        for i in range(len(L)):
                key = L[i]
                img = Image.open("{0}/img_align_celeba_png/{1}".format(celebA_path,key[:-4]+".png"))
                img = img.resize((128,128), PIL.Image.ANTIALIAS)
                if type =="test" :
                        img.save("{0}/testset/{1}".format(output_path, key[:-4]+".png"))
                #else:
                        #img.save("{0}/trainset/{1}".format(output_path, key[:-4]+".png"))

                print("{0} {1}/{2} done".format(type,i,len(L)))
