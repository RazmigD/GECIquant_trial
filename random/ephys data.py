from readimc import MCDFile, TXTFile

with MCDFile("/Users/razmigderounian/PycharmProjects/GECIquant_trial/160408_slice1_0006_lp499_ds1000_H5.mcd") as f:
    num_slides = len(f.slides)


