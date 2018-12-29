# coding: utf-8
import os 



if __name__ == '__main__':
    featurenlist = list(range(1, 4))
    sentnlist = list(range(3, 8))
    memlist = list(range(1, 5))
    droplist = list(range(1, 5))
    all_list = [[m * 20000, n * 50, o * 50, p * 0.2] for m in featurenlist for n in sentnlist for o in memlist for p in
                droplist if n * 50 > o * 50]
    print(len(all_list))
    print(all_list)

    for i in all_list:
        print("outbatch: *************************************************************")
        print("outbatch: " + str(i))
        os.system("python 06toxic_single.py " + str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + " " + str(i[3]))
#        model = main_loop(i[0], i[1], i[2], i[3])

